header = """
┌────────────────────────────────────────────┐
│              SHOPPING MART                 │
│        Market Basket Analysis Tool         │
└────────────────────────────────────────────┘"""

commands = """----------------------------------------------
COMMANDS

1. sup item[,item]                         # calcuates support #

2. con item[,item] --> item[,item]         # calculates confidence #

3. exit                                    # quits the application #
"""

def display_main_menu():
    print(header)

    error_message = None

    while True:
        print(f"\n{commands}")

        if error_message:
            print(error_message)
            error_message = None

        command = input("\nEnter command: ").strip().lower()

        if command == "exit":
            print("\nExiting Shopping Mart...")
            break

        elif command.startswith("sup"):
            print("SUP command")

        elif command.startswith("con"):
            print("CON command")

        else:
            error_message = "Invalid command. Try: sup, con, exit"

display_main_menu()