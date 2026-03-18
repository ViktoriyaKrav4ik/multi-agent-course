from agent import ResearchAgent


def main() -> None:
    print("Research Agent (HW4, custom ReAct loop). Type 'exit' to quit.")
    print("-" * 60)

    agent = ResearchAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            return
        if user_input.lower() in ("reset", "/reset"):
            agent.reset()
            print("Session memory cleared.")
            continue

        answer = agent.run(user_input)
        if answer:
            print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()

