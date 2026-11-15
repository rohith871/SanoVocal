from datasets import load_dataset

bharat = load_dataset("Banothuarpitha/bharatschemes-v1")
print(bharat)
print(bharat["train"].column_names)
print(bharat["train"][0])
