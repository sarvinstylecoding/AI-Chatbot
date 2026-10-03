import dotenv
import os
from openai import OpenAI
dotenv.load_dotenv()


class ChatBOT:
    def __init__(self):
        self.llm = OpenAI(api_key=os.getenv("GROQ_APIKEY"),
                          base_url="https://api.groq.com/openai/v1")
        self.messages = [{"role": "system", "content": """
            You are a helpful assistant,
            Your name is sarvinBot,
            Give me short and concise answer.
            Introduce your in first response.
            """}]
        self.total_tokens = 0

    def reply(self):
        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=self.messages,
            temperature=0.7,
            max_completion_tokens=255
        )

        response_content = response.choices[0].message.content

        self.messages.append(
            {"role": "assistant", "content":  response_content}
        )
        print("Input:", response.usage.prompt_tokens)
        print("Output:", response.usage.completion_tokens)
        print("Total:", response.usage.total_tokens)
        return response_content


    def chat(self):
        while True:
            user_input = input("your question :")
            self.messages.append({
                "role": "user",
                "content": user_input})
            if user_input.strip().lower() == "q":
                break
            answer = self.reply()
            print( f'\nSarvinBot: {answer}')


def main():
    chatbot = ChatBOT()
    chatbot.chat()


if __name__ == '__main__':
    main()
