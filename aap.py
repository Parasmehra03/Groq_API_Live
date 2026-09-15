from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from dotenv import load_dotenv
import streamlit as st
import os


# -----------------------------------------
# 1. Load Environment Variables
# -----------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# -----------------------------------------
# 2. Create the LLM
# -----------------------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=groq_api_key
)


# -----------------------------------------
# 3. Create Role + Few-Shot Prompt
# -----------------------------------------

messages = [

    # ROLE
    SystemMessage(
        content="""
You are a friendly Nutrition Assistant.

Your job is to help users understand whether
a food choice is generally healthy.

Classify food into one of these categories:

Healthy
Moderate
Avoid

Always respond using this format:

Category: <category>

Reason:
<short explanation>

Suggestion:
<short suggestion>

Keep your answer simple and suitable for beginners.
"""
    ),

    # -----------------------------------------
    # FEW-SHOT EXAMPLE 1
    # -----------------------------------------

    HumanMessage(
        content="I want to eat an apple."
    ),

    AIMessage(
        content="""
Category: Healthy

Reason:
Apple contains fiber, vitamins and natural nutrients.

Suggestion:
Apple can be included as part of a balanced diet.
"""
    ),

    # -----------------------------------------
    # FEW-SHOT EXAMPLE 2
    # -----------------------------------------

    HumanMessage(
        content="I want to eat pizza."
    ),

    AIMessage(
        content="""
Category: Moderate

Reason:
Pizza can contain a lot of refined carbohydrates,
cheese, salt and calories depending on the preparation.

Suggestion:
Eat it occasionally and prefer vegetables and
moderate cheese as toppings.
"""
    ),

    # -----------------------------------------
    # FEW-SHOT EXAMPLE 3
    # -----------------------------------------

    HumanMessage(
        content="I want to drink sugary soda every day."
    ),

    AIMessage(
        content="""
Category: Avoid

Reason:
Sugary drinks can contain a large amount of
added sugar and provide little nutritional value.

Suggestion:
Prefer water or unsweetened drinks for regular use.
"""
    )
]


# -----------------------------------------
# 4. Streamlit UI
# -----------------------------------------

st.title("🥗 Your Personal Healthy Food Assistant")

st.write(
    "Ask me whether a food or drink is generally "
    "Healthy, Moderate, or Avoid."
)


# -----------------------------------------
# 5. Get User's Question
# -----------------------------------------

food = st.chat_input(
    "What food or drink are you thinking about?"
)


if food:

    # Display user's message
    st.chat_message("user").write(food)

    # Add user's actual question
    messages.append(
        HumanMessage(
            content=f"I want to eat or drink {food}."
        )
    )

    # -----------------------------------------
    # 6. Send Request to Groq
    # -----------------------------------------

    with st.spinner("Thinking..."):

        response = model.invoke(messages)

    # -----------------------------------------
    # 7. Display AI Response
    # -----------------------------------------

    st.chat_message("assistant").write(
        response.content
    )
