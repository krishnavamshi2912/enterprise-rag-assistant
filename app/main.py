from app.retrieval.pipeline import ask, build_teleco_assistant

def main():
    print("Building the teleco knowledge assistant")

    agent = build_teleco_assistant()

    print("Assistant ready....")

    question = "What is billing?"
    answer = ask(agent, question)

    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()