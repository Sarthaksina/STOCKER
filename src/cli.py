"""
CLI entry point for STOCKER: Interactive stock analytics, portfolio planning, risk analysis, and event detection.
"""
import sys
from src.entity.config_entity import StockerConfig
from src.features.agent import StockerAgent
import pprint

def print_menu():
    print("\n==== STOCKER CLI ====")
    print("1. Portfolio Recommendation")
    print("2. Holdings Analysis")
    print("3. News & Sentiment")
    print("4. Earnings Call Summary")
    print("5. Peer Comparison")
    print("6. Event & Macro Event Detection")
    print("7. Exit")

def get_user_info():
    print("\nEnter your details for personalized planning:")
    age = int(input("Age: "))
    risk = input("Risk Appetite (conservative/moderate/aggressive): ").strip().lower()
    sip = float(input("Monthly SIP amount (0 if none): "))
    lumpsum = float(input("Lumpsum investment (0 if none): "))
    years = int(input("Investment horizon (years): "))
    return {
        "age": age,
        "risk_appetite": risk,
        "sip_amount": sip if sip > 0 else None,
        "lumpsum": lumpsum if lumpsum > 0 else None,
        "years": years
    }

def main():
    config = StockerConfig()
    agent = StockerAgent(config)
    while True:
        print_menu()
        choice = input("Choose an option: ").strip()
        if choice == "1":
            user_info = get_user_info()
            result = agent.answer("portfolio", user_info)
            pprint.pprint(result)
        elif choice == "2":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"holdings {symbol}")
            pprint.pprint(result)
        elif choice == "3":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"news {symbol}")
            pprint.pprint(result)
        elif choice == "4":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"concall {symbol}")
            pprint.pprint(result)
        elif choice == "5":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"peer {symbol}")
            pprint.pprint(result)
        elif choice == "6":
            symbol = input("Enter stock symbol: ").strip().upper()
            result = agent.answer(f"event {symbol}")
            pprint.pprint(result)
        elif choice == "7":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
