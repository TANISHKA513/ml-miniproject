import pandas as pd

# Load the original dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Original Dataset Shape:", data.shape)

# Remove duplicate messages
data = data.drop_duplicates()

print("Dataset Shape After Removing Duplicates:", data.shape)

# Save cleaned dataset
data.to_csv("cleaned_sms_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")