import base64
import time
from http import HTTPStatus
from uuid import uuid4

from aleph.sdk import AuthenticatedAlephHttpClient
from aleph.sdk.chains.ethereum import ETHAccount
from aleph_message.models.execution import Encoding
from ecies import encrypt, decrypt
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.middleware.cors import CORSMiddleware

from src.config import config
from src.interfaces.agent import (
    Agent,
    SetupAgentBody,
    DeleteAgentBody,
    UpdateAgentResponse,
)
from src.interfaces.aleph import AlephVolume
from src.utils.agent import fetch_agents, fetch_agent_program_message
from src.utils.storage import upload_file

app = FastAPI(title="LibertAI agents")

origins = [
    "https://chat.libertai.io",
    "http://localhost:9000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/agent", description="Setup a new agent on subscription")
async def setup(body: SetupAgentBody) -> None:
    agent_id = str(uuid4())

    secret = str(uuid4())
    # Encrypting the secret ID with our public key
    encrypted_secret = encrypt(config.ALEPH_SENDER_PK, secret.encode())
    # Encoding it in base64 to avoid data loss when stored on Aleph
    base64_encrypted_secret = base64.b64encode(encrypted_secret).decode()

    agent = Agent(
        id=agent_id,
        subscription_id=body.subscription_id,
        vm_hash=None,
        encrypted_secret=base64_encrypted_secret,
        last_update=int(time.time()),
        tags=[agent_id, body.subscription_id, body.account.address],
    )

    aleph_account = ETHAccount(config.ALEPH_SENDER_SK)
    async with AuthenticatedAlephHttpClient(
        account=aleph_account, api_server=config.ALEPH_API_URL
    ) as client:
        post_message, _ = await client.create_post(
            post_content=agent.dict(),
            post_type=config.ALEPH_AGENT_POST_TYPE,
            channel=config.ALEPH_CHANNEL,
        )


@app.put("/agent", description="Deploy an agent or update it")
async def update(
    agent_id: str = Form(),
    secret: str = Form(),
    code: UploadFile = File(...),
    packages: UploadFile = File(...),
) -> UpdateAgentResponse:
    agents = await fetch_agents([agent_id])

    if len(agents) != 1:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found.",
        )
    agent = agents[0]
    agent_program = (
        await fetch_agent_program_message(agent.vm_hash)
        if agent.vm_hash is not None
        else None
    )

    # Decode the base64 secret
    encrypted_secret = base64.b64decode(agent.encrypted_secret)

    decrypted_secret = decrypt(config.ALEPH_SENDER_SK, encrypted_secret).decode()
    if secret != decrypted_secret:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="The secret provided doesn't match the one of this agent.",
        )

    previous_code_ref = (
        agent_program.content.code.ref if agent_program is not None else None
    )
    # TODO: additional checks on the type of volume, find the right one based on mount etc
    previous_packages_ref = (
        agent_program.content.volumes[0].ref if agent_program is not None else None  # type: ignore
    )

    code_ref = await upload_file(code, previous_code_ref)
    packages_ref = await upload_file(packages, previous_packages_ref)

    if agent_program is not None:
        # Program is already deployed and we updated the volumes, exiting here
        return UpdateAgentResponse(vm_hash=agent_program.item_hash)

    # Register the program
    aleph_account = ETHAccount(config.ALEPH_SENDER_SK)
    async with AuthenticatedAlephHttpClient(
        account=aleph_account, api_server=config.ALEPH_API_URL
    ) as client:
        message, _ = await client.create_program(
            program_ref=code_ref,
            entrypoint="run",
            runtime="63f07193e6ee9d207b7d1fcf8286f9aee34e6f12f101d2ec77c1229f92964696",
            channel=config.ALEPH_CHANNEL,
            encoding=Encoding.squashfs,
            persistent=False,
            volumes=[
                AlephVolume(
                    comment="Python packages",
                    mount="/opt/packages",
                    ref=packages_ref,
                    use_latest=True,
                ).dict()
            ],
        )

        # Updating the related POST message
        await client.create_post(
            post_content=Agent(
                **agent.dict(exclude={"vm_hash", "last_update"}),
                vm_hash=message.item_hash,
                last_update=int(time.time()),
            ),
            post_type="amend",
            ref=agent.post_hash,
            channel=config.ALEPH_CHANNEL,
        )
    return UpdateAgentResponse(vm_hash=message.item_hash)


@app.delete("/agent", description="Remove an agent on subscription end")
async def delete(body: DeleteAgentBody):
    # TODO
    pass
