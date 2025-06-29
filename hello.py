import os
import chainlit as cl
from agents import Agent, RunConfig,AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

#Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",

)

#Step 2: MOdel
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

agent1 = Agent(
    instructions="You are a helpful assistant that answer questions and tasks.",
    name="Apna Assistant"
)


result = Runner.run_sync(
    agent1,
    input="What is the Capital of Pakistan",
    run_config=run_config,
)

print(result.final_output)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello I'm Apna Assistant. How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content}) 

    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            # print(event.data.delta, end="", flush=True)

            await msg.stream_token(event.data.delta)


    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    # await cl.Message(content=result.final_output).send()

