import os
def save_training_history(history, file_name, save_dir="training_logs"):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, "w") as f:
        f.write(repr(history))  # Save the exact structure in a Python-readable format



def load_training_history(file_name):
    with open(file_name, "r") as f:
        return eval(f.read())  # Reconstruct the exact structure using eval()

def check_for_overwrites(model_dir, epochs):
    if not (isinstance(epochs, tuple) and len(epochs) == 2):
        print(epochs, type(epochs))
        print("Please provide a tuple of the form: (start_epoch, end_epoch-1")
        return 0
    last_model_epoch = None

    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            try:
                number_str = filename.split('_')[-1].replace('.pth', '')
                number = int(number_str)

                if last_model_epoch is None or number > last_model_epoch:
                    last_model_epoch = number
            except ValueError:
                continue

    if last_model_epoch >= epochs[0] and last_model_epoch is not None:
        print(f"The model has already trained for more than {epochs[0]} epochs. Please start at epoch {last_model_epoch+1}.")
        return 0


    return 1