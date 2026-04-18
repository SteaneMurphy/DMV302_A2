from algorithms.association_core import preprocess_data, calculate_support, calculate_confidence
from utility import parse_sup_command, parse_con_command, display_result, header, commands

def main_menu():
    print(header)

    error_message = None

    while True:                                                                 # keep application running until exit
        print(f"\n{commands}")                                                  # display command options text

        if error_message:                                                       # print error message if error
            print(error_message)
            error_message = None

        command = input("\nEnter command: ").strip().lower()                    # parse user input

        if command == "exit":                                                   # exit applicatoin
            print("\nExiting Shopping Mart...")
            break

        elif command.startswith("sup"):                                         # support menu option
            itemset = parse_sup_command(command)                                # parse input for support function
            support = calculate_support(itemset, DATA)                          # calculate support value
            display_result(                                                     # display and print result
                itemset, 
                support, 
                association_type="support"
            )
    
        elif command.startswith("con"):                                         # confidence menu option
            if "-->" not in command:                                            # validation (-->)
                error_message = "Invalid format. Use: con item --> item"

            itemsetA, itemsetB = parse_con_command(command)                     # parse input for confidence function
            confidence = calculate_confidence(itemsetA, itemsetB, DATA)         # calculate confidence value
            display_result(                                                     # display and print result
                itemsetA, 
                confidence, 
                association_type="confidence", 
                itemset_b=itemsetB
            )

        else:
            error_message = "Invalid command. Try: sup, con, exit"

DATA = preprocess_data("data/shopping_transactions.csv")                        # preprocess and load dataset (application state)
main_menu()                                                                     