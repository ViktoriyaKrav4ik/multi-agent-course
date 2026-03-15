from langchain_core.messages import HumanMessage

from agent import agent

# Щоб агент пам'ятав діалог у межах сесії (вимога домашки 3)
CONFIG = {"configurable": {"thread_id": "default"}}


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
        )
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content") and last.content:
                print(f"\nAgent: {last.content}")


if __name__ == "__main__":
    main()
