import pandas as pd
from g4f.client import Client
import re

# Membaca dataset
df = pd.read_csv('data_sample_50.csv')

# Daftar label valid (pastikan semuanya case-sensitive dan konsisten)
labels = [
    'Information Technology', 'Data Science', 'Database', 'DevOps',
    'DotNet Developer', 'ETL Developer', 'Java Developer', 'Network Security Engineer',
    'Python Developer', 'React Developer', 'SAP Developer',
    'SQL Developer', 'Testing', 'Web Designing'
]

# Fungsi preprocessing teks deskripsi pekerjaan
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'please note.*?deloitte', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www.\S+', '', text)               # Hapus URL
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)                # Hapus karakter spesial
    text = re.sub(r'\s+', ' ', text).strip()                  # Normalisasi spasi
    return text

# Inisialisasi GPT client
client = Client()

# Prompt klasifikasi yang diperketat
def classify_with_gpt(text):
    try:
        prompt = f"""
You are an expert in job classification. Given the following job description, return only the relevant categories from the list below. Select all applicable categories, if any.

Categories (return only from this list):
{', '.join(labels)}

Guidelines:
- Choose only from the list above. Do NOT make up categories.
- Return multiple categories if relevant.
- If no specific category is suitable, ONLY then use 'Information Technology'.
- Return only the list of categories separated by commas. No extra text.

Job description:
\"\"\"
{text}
\"\"\"
        """.strip()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw_output = response.choices[0].message.content.strip()

        # Parsing dan validasi hasil
        predicted_labels = [label.strip() for label in raw_output.split(',')]
        valid_labels = list(set([label for label in predicted_labels if label in labels]))

        return valid_labels if valid_labels else ['Information Technology']
    
    except Exception as e:
        return [f"<Error: {str(e)}>"]

# Terapkan preprocessing dan klasifikasi
df['job_cleaned'] = df['job description'].fillna('').apply(preprocess_text)
df['predicted_label'] = df['job_cleaned'].apply(classify_with_gpt)
df['predicted_label'] = df['predicted_label'].apply(lambda x: ', '.join(x))

# Simpan ke file
df.to_csv('data_final_labelled.csv', index=False)
df.to_excel('data_final_labelled.xlsx', index=False)

print("✅ Proses klasifikasi selesai.")
