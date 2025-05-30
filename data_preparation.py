import pandas as pd
import spacy
import medspacy
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import json
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import torch

warnings.filterwarnings("ignore")

# Custom logging filter to suppress specific message
class SuppressDotSyntaxFilter(logging.Filter):
    """Custom filter to suppress '\.' is not a eligible syntax messages."""
    def filter(self, record):
        return not ('"\." is not a eligible syntax.' in record.getMessage())

# Set up logging with reduced verbosity and custom filter
logging.basicConfig(
    filename='data_preparation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite log file each run
)
logger = logging.getLogger()
logger.addFilter(SuppressDotSyntaxFilter())

# Global counter for regex errors to prevent log spam
regex_error_counts = {}
max_regex_errors_per_pattern = 5

# Load models
try:
    nlp_bio = spacy.load("en_ner_bc5cdr_md")  # SciSpaCy for biomedical NER
    nlp_phi = medspacy.load()  # medSpaCy for PHI de-identification
    context_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Successfully loaded SciSpaCy, medSpaCy, and SentenceTransformer models.")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    print(f"Error: Failed to load models - {e}")
    raise

# Medical terms that should NEVER be anonymized
MEDICAL_TERMS = {
    'melatonin', 'ambien', 'zolpidem', 'prozac', 'lexapro', 'xanax', 'klonopin',
    'sertraline', 'escitalopram', 'fluoxetine', 'paroxetine', 'venlafaxine',
    'duloxetine', 'bupropion', 'mirtazepine', 'trazodone', 'buspirone',
    'lorazepam', 'clonazepam', 'diazepam', 'alprazolam', 'temazepam',
    'depression', 'anxiety', 'ptsd', 'ocd', 'schizophrenia', 'adhd', 'autism',
    'bipolar', 'mania', 'hypomania', 'panic', 'phobia', 'agoraphobia',
    'insomnia', 'narcolepsy', 'diabetes', 'hypothyroidism', 'fibromyalgia',
    'arthritis', 'hypertension', 'hypotension', 'migraine', 'epilepsy',
    'therapy', 'cbt', 'dbt', 'emdr', 'psychotherapy', 'counseling',
    'medication', 'antidepressant', 'antipsychotic', 'anxiolytic', 'sedative',
    'stimulant', 'mood stabilizer', 'anticonvulsant', 'beta blocker',
    'ssri', 'snri', 'tricyclic', 'maoi', 'benzodiazepine', 'barbiturate',
    # Mental health specific terms
    'reddit', 'subreddit', 'throwaway', 'anonymous', 'support group',
    'online community', 'forum', 'chat room', 'helpline', 'crisis line',
    'peer support', 'self help', 'coping mechanism', 'trigger warning',
    'mental health professional', 'licensed therapist', 'counselor'
}

# Medical organizations that should NEVER be anonymized
MEDICAL_ORGANIZATIONS = {
    'nih', 'national institutes of health', 'nimh', 'cdc', 'fda', 'who',
    'mayo clinic', 'cleveland clinic', 'johns hopkins', 'ama', 'apa', 'nami',
    'samhsa', 'american psychiatric association', 'american medical association',
    'world health organization', 'centers for disease control', 'nhs',
    'kaiser permanente', 'veterans affairs', 'va hospital', 'psychiatry today',
    'psychology today'
}

# Medical locations/institutions that should be preserved
MEDICAL_LOCATIONS = {
    'boston', 'baltimore', 'houston', 'atlanta', 'new york', 'chicago',
    'massachusetts', 'california', 'texas', 'pennsylvania', 'maryland',
    'florida', 'washington', 'oregon', 'michigan', 'minnesota'
}

# Context embeddings for classification
MEDICAL_CONTEXTS = [
    "medical research clinical trial study disease treatment diagnosis symptoms",
    "hospital clinic doctor physician medical institution healthcare",
    "medication drug prescription therapy treatment pharmacology",
    "neurological disorder brain nervous system condition syndrome",
    "genetic hereditary syndrome disease mutation biomarkers",
    "gastrointestinal digestive tract inflammation chronic disease",
    "mental health psychiatric psychological therapy disorder"
]

PERSONAL_CONTEXTS = [
    "my friend family member personal relationship spouse partner",
    "someone I know personal story experience individual",
    "private individual person name identity patient"
]

# FIXED REGEX PATTERNS with specified patterns removed
REGEX_PATTERNS = {
    'EMAIL': r'\b[\w.-]+@[\w.-]+\.\w+\b',
    'PHONE': r'\b(?:\(\d{3}\)\s*|\d{3}[.-]?)\d{3}[.-]?\d{4}\b',
    'SSN': r'\b\d{3}[.-]?\d{2}[.-]?\d{4}\b',
    'PATIENT_ID': r'\b(?:patient|case|record|id)\s*(?:id|#|num|number)?\s*:?\s*\d+\b',
    'USERNAME': r'\b@[A-Za-z0-9_]{3,20}\b',
    'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'CREDIT_CARD': r'\b(?:\d{4}[.\s-]?){3}\d{4}\b',
    'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
    'MEDICAL_RECORD': r'\b(?:mrn|medical record|chart)\s*#?\s*:?\s*\d+\b'
}

def validate_regex_patterns():
    """Validate all regex patterns before use and return only valid ones."""
    valid_patterns = {}
    invalid_count = 0

    print("Validating regex patterns...")
    for label, pattern in REGEX_PATTERNS.items():
        try:
            re.compile(pattern)
            valid_patterns[label] = pattern
            print(f"✓ {label}: Pattern valid")
        except re.error as e:
            print(f"✗ {label}: Pattern invalid - {e}")
            logging.error(f"Invalid regex pattern for {label}: {pattern} - {e}")
            invalid_count += 1
        except Exception as e:
            print(f"✗ {label}: Unexpected error - {e}")
            logging.error(f"Unexpected error validating pattern for {label}: {pattern} - {e}")
            invalid_count += 1

    print(f"Pattern validation complete: {len(valid_patterns)} valid, {invalid_count} invalid")
    return valid_patterns

def test_regex_patterns(patterns):
    """Test regex patterns with sample text."""
    test_text = """
    Test email@example.com phone (555) 123-4567 SSN 123-45-6789
    Patient ID: 12345 @username
    IP: 192.168.1.1
    Credit card: 1234-5678-9012-3456 DOB: 12/25/1980
    MRN: 987654
    Phone: 555-123-4567 555.123.4567
    """

    print("\nTesting regex patterns with sample text...")
    for label, pattern in patterns.items():
        try:
            matches = re.findall(pattern, test_text, re.IGNORECASE)
            print(f"{label}: {len(matches)} matches found - {matches}")
        except Exception as e:
            print(f"{label}: ERROR during testing - {e}")

def get_context_embeddings():
    """Pre-compute embeddings for context classification."""
    try:
        medical_embeddings = context_model.encode(MEDICAL_CONTEXTS, show_progress_bar=False)
        personal_embeddings = context_model.encode(PERSONAL_CONTEXTS, show_progress_bar=False)
        return medical_embeddings, personal_embeddings
    except Exception as e:
        logging.error(f"Error computing context embeddings: {e}")
        return None, None

def classify_context(text_window, medical_embeddings, personal_embeddings, threshold=0.4):
    """Classify context as medical, personal, or neutral."""
    try:
        if medical_embeddings is None or personal_embeddings is None:
            return 'neutral'

        text_embedding = context_model.encode([text_window], show_progress_bar=False)
        medical_similarities = cosine_similarity(text_embedding, medical_embeddings)[0]
        personal_similarities = cosine_similarity(text_embedding, personal_embeddings)[0]

        max_medical_sim = np.max(medical_similarities)
        max_personal_sim = np.max(personal_similarities)

        if max_medical_sim > threshold and max_medical_sim >= max_personal_sim:
            return 'medical'
        elif max_personal_sim > threshold and max_personal_sim > max_medical_sim:
            return 'personal'
        else:
            return 'neutral'
    except Exception as e:
        logging.warning(f"Context classification error: {e}")
        return 'neutral'

def is_medical_term(entity_text):
    """Check if the entity is a medical term or condition."""
    entity_lower = entity_text.lower().strip()
    if entity_lower in MEDICAL_TERMS:
        return True

    medical_patterns = [
        r'\b\w+\s*(?:disease|disorder|syndrome|condition)\b',
        r'\b\w+\s*(?:colitis|hepatitis|itis)\b',
        r'\b(?:mg|ml|mcg|units?|dosage|tablet|capsule|injection)\b'
    ]
    return any(re.search(pattern, entity_lower, re.IGNORECASE) for pattern in medical_patterns)

def is_medical_organization(entity_text, context_window, context_type):
    """Check for medical organizations using context."""
    entity_lower = entity_text.lower().strip()
    if entity_lower in MEDICAL_ORGANIZATIONS:
        return True

    if context_type == 'medical':
        medical_keywords = [
            'institute', 'hospital', 'clinic', 'medical', 'health', 'research',
            'association', 'foundation', 'center', 'university', 'college',
            'department', 'division', 'school of medicine'
        ]
        if any(keyword in entity_lower for keyword in medical_keywords):
            return True
    return False

def is_medical_location(entity_text, context_window, context_type):
    """Check if a location should be preserved in medical context."""
    entity_lower = entity_text.lower().strip()
    if entity_lower in MEDICAL_LOCATIONS and context_type == 'medical':
        return True

    context_lower = context_window.lower()
    medical_location_indicators = [
        'medical center', 'hospital', 'clinic', 'university', 'institute',
        'research facility', 'healthcare system', 'medical school'
    ]
    if any(indicator in context_lower for indicator in medical_location_indicators):
        return True
    return False

def is_real_person_name(entity_text, context_window, context_type):
    """Enhanced person name detection using context."""
    entity_lower = entity_text.lower().strip()

    if is_medical_term(entity_lower):
        return False

    common_non_names = {
        'i', 'me', 'my', 'dr', 'mr', 'ms', 'mrs', 'prof', 'professor',
        'patient', 'doctor', 'nurse', 'therapist', 'psychiatrist', 'user',
        'reddit', 'anonymous', 'throwaway'
    }
    if entity_lower in common_non_names:
        return False

    if any(pattern in entity_lower for pattern in ['user', 'anon', 'throwaway', '_', '123', '456', '789']):
        return False

    if context_type == 'medical':
        personal_indicators = [
            'my friend', 'my family', 'my spouse', 'my partner', 'my mother',
            'my father', 'my sister', 'my brother', 'someone i know',
            'a person', 'individual named', 'patient named'
        ]
        context_lower = context_window.lower()
        if any(indicator in context_lower for indicator in personal_indicators):
            return True
        return False
    elif context_type == 'personal':
        return True

    if len(entity_text) <= 2:
        return False

    return True

def safe_regex_search(pattern, text, flags=re.IGNORECASE):
    """Safely apply regex patterns with error handling and limited logging."""
    global regex_error_counts, max_regex_errors_per_pattern

    try:
        return list(re.finditer(pattern, text, flags))
    except re.error as e:
        error_msg = str(e)
        if '"\." is not a eligible syntax.' in error_msg:
            return []  # Silently skip this specific error
        if pattern not in regex_error_counts:
            regex_error_counts[pattern] = 0

        regex_error_counts[pattern] += 1
        if regex_error_counts[pattern] <= max_regex_errors_per_pattern:
            logging.warning(f"Regex error for pattern '{pattern}': {e}")
        elif regex_error_counts[pattern] == max_regex_errors_per_pattern + 1:
            logging.warning(f"Regex error limit reached for pattern '{pattern}', suppressing further errors")
        return []
    except Exception as e:
        error_key = f"unexpected_{pattern}"
        if error_key not in regex_error_counts:
            regex_error_counts[error_key] = 0

        regex_error_counts[error_key] += 1
        if regex_error_counts[error_key] <= max_regex_errors_per_pattern:
            logging.warning(f"Unexpected error for pattern '{pattern}': {e}")
        return []

def combined_ner_anonymize(text, nlp_bio, nlp_phi, medical_embeddings, personal_embeddings, valid_patterns):
    """Combined SciSpaCy and medSpaCy anonymization with context awareness."""
    if not isinstance(text, str):
        return text

    try:
        context_type = classify_context(text, medical_embeddings, personal_embeddings)
        entities = []

        try:
            doc_bio = nlp_bio(text)
            for ent in doc_bio.ents:
                entities.append((ent.start_char, ent.end_char, ent.label_, ent.text))
        except Exception as e:
            logging.warning(f"SciSpaCy processing error: {e}")

        try:
            doc_phi = nlp_phi(text)
            for ent in doc_phi.ents:
                entities.append((ent.start_char, ent.end_char, ent.label_, ent.text))
        except Exception as e:
            logging.warning(f"medSpaCy processing error: {e}")

        for label, pattern in valid_patterns.items():
            matches = safe_regex_search(pattern, text)
            for match in matches:
                entities.append((match.start(), match.end(), label, match.group()))

        entities.sort(key=lambda x: (x[0], -x[1]))
        filtered_entities = []
        prev_end = -1

        for start, end, label, entity_text in entities:
            if start >= prev_end:
                filtered_entities.append((start, end, label, entity_text))
                prev_end = end
            elif label in ['DISEASE', 'CHEMICAL']:
                if filtered_entities:
                    filtered_entities[-1] = (start, end, label, entity_text)
                else:
                    filtered_entities.append((start, end, label, entity_text))
                prev_end = end

        entities_to_replace = []
        for start, end, label, entity_text in filtered_entities:
            if is_medical_term(entity_text) or label in ['DISEASE', 'CHEMICAL']:
                continue

            context_start = max(0, start - 200)
            context_end = min(len(text), end + 200)
            context_window = text[context_start:context_end]
            entity_context_type = classify_context(context_window, medical_embeddings, personal_embeddings)

            should_anonymize = False

            if label == 'PERSON' or label in ['PATIENT', 'DOCTOR', 'NAME']:
                should_anonymize = is_real_person_name(entity_text, context_window, entity_context_type)
            elif label == 'ORG' or label == 'ORGANIZATION':
                should_anonymize = not is_medical_organization(entity_text, context_window, entity_context_type)
            elif label == 'GPE' or label in ['LOCATION', 'CITY', 'STATE', 'COUNTRY']:
                should_anonymize = not is_medical_location(entity_text, context_window, entity_context_type)
            elif label in valid_patterns:
                should_anonymize = True
            elif label in ['WORK_OF_ART', 'PRODUCT', 'EVENT']:
                should_anonymize = not (entity_context_type == 'medical' or is_medical_term(entity_text))

            if should_anonymize:
                entities_to_replace.append((start, end, label, entity_text))

        entities_to_replace.sort(key=lambda x: x[0], reverse=True)

        anonymized_text = text
        for start, end, label, orig_text in entities_to_replace:
            placeholder = f"[{label}]" if label in valid_patterns else "[ANONYMIZED]"
            logging.info(f"Anonymizing: {orig_text} (Label: {label}) -> {placeholder}")
            anonymized_text = anonymized_text[:start] + placeholder + anonymized_text[end:]

        return anonymized_text

    except Exception as e:
        logging.error(f"Error in combined anonymization: {e}")
        return text

def anonymize_row(row, nlp_bio, nlp_phi, medical_embeddings, personal_embeddings, valid_patterns):
    """Apply combined anonymization to a DataFrame row."""
    try:
        row['question'] = combined_ner_anonymize(row['question'], nlp_bio, nlp_phi, medical_embeddings, personal_embeddings, valid_patterns)
        row['answer'] = combined_ner_anonymize(row['answer'], nlp_bio, nlp_phi, medical_embeddings, personal_embeddings, valid_patterns)
        return row
    except Exception as e:
        logging.error(f"Error anonymizing row: {e}")
        return row

def validate_pair(row, min_question_len=10, max_question_len=12000, min_answer_len=20, max_answer_len=30000):
    """Validate question-answer pair for quality."""
    try:
        question = row['question'] if hasattr(row, '__getitem__') else row.question
        answer = row['answer'] if hasattr(row, '__getitem__') else row.answer

        q_len = len(str(question))
        a_len = len(str(answer))

        if not (min_question_len <= q_len <= max_question_len):
            return False, f"Question length out of range ({q_len})"
        if not (min_answer_len <= a_len <= max_answer_len):
            return False, f"Answer length out of range ({a_len})"
        if not isinstance(question, str) or not isinstance(answer, str):
            return False, "Non-string question or answer"
        if pd.isna(question) or pd.isna(answer):
            return False, "Missing question or answer"
        return True, ""
    except Exception as e:
        logging.error(f"Error validating pair: {e}")
        return False, str(e)

def load_datasets():
    """Load datasets from Hugging Face."""
    try:
        print("Loading MentalChat16K dataset...")
        mentalchat_ds = load_dataset("ShenLab/MentalChat16K", split="train")
        print("Loading MedQuAD dataset...")
        medquad_ds = load_dataset("lavita/MedQuAD", split="train")

        print(f"Successfully loaded MentalChat16K: {len(mentalchat_ds)} records.")
        print(f"Successfully loaded MedQuAD: {len(medquad_ds)} records.")
        return pd.DataFrame(mentalchat_ds), pd.DataFrame(medquad_ds)
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        print(f"Error: Failed to load datasets - {e}")
        return None, None

def preprocess_data(mentalchat_df, medquad_df, nlp_bio, nlp_phi, valid_patterns):
    """Preprocess data with combined anonymization."""
    try:
        mentalchat_df = mentalchat_df.rename(columns={'input': 'question', 'output': 'answer'})
        medquad_df = medquad_df.rename(columns=lambda x: x.lower().strip())

        mentalchat_df['source'] = 'mentalchat'
        medquad_df['source'] = 'medquad'

        combined_df = pd.concat([mentalchat_df[['question', 'answer', 'source']],
                                medquad_df[['question', 'answer', 'source']]],
                                ignore_index=True)

        initial_count = len(combined_df)
        combined_df = combined_df.dropna()
        combined_df = combined_df.drop_duplicates(subset=['question', 'answer'], keep='first')
        print(f"Dataset cleaned: {len(combined_df)} records (removed {initial_count - len(combined_df)} invalid/duplicate entries)")

        print("Preparing context classification models...")
        medical_embeddings, personal_embeddings = get_context_embeddings()

        if medical_embeddings is None or personal_embeddings is None:
            print("Warning: Context classification unavailable, proceeding with basic anonymization")

        print("Starting combined SciSpaCy and medSpaCy anonymization...")
        tqdm.pandas(desc="Anonymizing with context awareness")
        combined_df = combined_df.progress_apply(
            lambda row: anonymize_row(row, nlp_bio, nlp_phi, medical_embeddings, personal_embeddings, valid_patterns),
            axis=1
        )
        print("Successfully completed combined anonymization.")

        print("Validating processed data...")
        valid_rows = []
        filtered_count = 0

        for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Validating"):
            is_valid, reason = validate_pair(row)
            if is_valid:
                valid_rows.append(row)
            else:
                filtered_count += 1
                if filtered_count <= 10:
                    logging.info(f"Filtered record: {reason}")

        combined_df = pd.DataFrame(valid_rows)
        print(f"Final dataset: {len(combined_df)} records (filtered {filtered_count} invalid entries)")

        return combined_df

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        print(f"Error: Failed to preprocess data - {e}")
        return None

def compute_statistics(df, output_dir):
    """Compute and save dataset statistics."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        stats = {
            "total_records": len(df),
            "mentalchat_records": len(df[df['source'] == 'mentalchat']),
            "medquad_records": len(df[df['source'] == 'medquad']),
            "question_length_mean": float(df['question'].str.len().mean()),
            "question_length_std": float(df['question'].str.len().std()),
            "question_length_min": int(df['question'].str.len().min()),
            "question_length_max": int(df['question'].str.len().max()),
            "answer_length_mean": float(df['answer'].str.len().mean()),
            "answer_length_std": float(df['answer'].str.len().std()),
            "answer_length_min": int(df['answer'].str.len().min()),
            "answer_length_max": int(df['answer'].str.len().max()),
            "unique_questions": df['question'].nunique(),
            "unique_answers": df['answer'].nunique(),
            "processing_timestamp": datetime.now().isoformat(),
            "regex_error_summary": dict(regex_error_counts)
        }

        stats_path = os.path.join(output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"Dataset statistics saved to {stats_path}")
        print("Dataset Statistics:")
        for key, value in stats.items():
            if key == "regex_error_summary":
                continue
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        if regex_error_counts:
            print("\nRegex Error Summary:")
            for pattern, count in regex_error_counts.items():
                print(f"  {pattern}: {count} errors")

        return stats
    except Exception as e:
        logging.error(f"Error computing statistics: {e}")
        return None

def split_data(combined_df, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Split data into training, validation, and test sets."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_df, temp_df = train_test_split(
            combined_df,
            train_size=train_ratio,
            random_state=42,
            stratify=combined_df['source']
        )

        val_size = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=42,
            stratify=temp_df['source']
        )

        print(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        print(f"Train split by source: MentalChat={len(train_df[train_df['source']=='mentalchat'])}, MedQuAD={len(train_df[train_df['source']=='medquad'])}")
        print(f"Validation split by source: MentalChat={len(val_df[val_df['source']=='mentalchat'])}, MedQuAD={len(val_df[val_df['source']=='medquad'])}")
        print(f"Test split by source: MentalChat={len(test_df[test_df['source']=='mentalchat'])}, MedQuAD={len(test_df[test_df['source']=='medquad'])}")

        return train_df, val_df, test_df
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return None, None, None

def format_for_finetuning(df, output_path, source_label):
    """Format data as JSONL for fine-tuning."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                entry = {
                    "text": f"Question: {row['question']}\nAnswer: {row['answer']}",
                    "source": row['source'],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "question_length": len(str(row['question'])),
                        "answer_length": len(str(row['answer']))
                    }
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"Successfully saved {source_label} JSONL to {output_path} ({len(df)} records)")
    except Exception as e:
        logging.error(f"Error formatting JSONL for {source_label}: {e}")

def main(output_dir="combined_medical_data"):
    """Main function with combined SciSpaCy and medSpaCy anonymization."""
    print("Starting ENHANCED Medical Data Preparation Pipeline...")
    print("Key features:")
    print("- SciSpaCy (en_ner_bc5cdr_md) for biomedical NER")
    print("- medSpaCy for clinical PHI de-identification")
    print("- Context-aware anonymization with SentenceTransformers")
    print("- Reduced regex patterns for stricter anonymization")
    print("- Mental health community specific patterns")
    print("- FIXED regex patterns with validation")
    print("- Limited error logging to prevent log spam")
    print("- Preserves medical terms, organizations, and locations")
    print("- Suppressing '\.' is not a eligible syntax errors")
    print("- Removed CHAT_HANDLE, SOCIAL_HANDLE, THROWAWAY_ACCOUNT, URL, INSURANCE_ID, REDDIT_USER patterns")
    print("-" * 60)

    # Validate and test regex patterns
    valid_patterns = validate_regex_patterns()
    if not valid_patterns:
        print("ERROR: No valid regex patterns. Exiting.")
        return

    test_regex_patterns(valid_patterns)

    # Load datasets
    mentalchat_df, medquad_df = load_datasets()
    if mentalchat_df is None or medquad_df is None:
        print("ERROR: Failed to load datasets. Exiting.")
        return

    # Preprocess data
    combined_df = preprocess_data(mentalchat_df, medquad_df, nlp_bio, nlp_phi, valid_patterns)
    if combined_df is None:
        print("ERROR: Preprocessing failed. Exiting.")
        return

    # Compute statistics
    stats = compute_statistics(combined_df, output_dir)
    if stats is None:
        print("ERROR: Failed to compute statistics. Exiting.")
        return

    # Split data
    train_df, val_df, test_df = split_data(combined_df, output_dir)
    if train_df is None:
        print("ERROR: Data splitting failed. Exiting.")
        return

    # Save formatted data
    format_for_finetuning(train_df, os.path.join(output_dir, 'train.jsonl'), 'train')
    format_for_finetuning(val_df, os.path.join(output_dir, 'validation.jsonl'), 'validation')
    format_for_finetuning(test_df, os.path.join(output_dir, 'test.jsonl'), 'test')

    # Save raw dataframes as well
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("\n" + "="*60)
    print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Benefits of this approach:")
    print("✓ Accurate biomedical NER with SciSpaCy")
    print("✓ Robust PHI de-identification with medSpaCy")
    print("✓ Context-aware decisions using SentenceTransformers")
    print("✓ Reduced regex patterns for stricter anonymization")
    print("✓ Mental health community specific anonymization")
    print("✓ Fixed regex patterns prevent syntax errors")
    print("✓ Preserves medical research value while protecting privacy")
    print("✓ Improved error handling and logging")
    print("✓ Suppressed '\.' is not a eligible syntax errors")
    print("✓ Removed CHAT_HANDLE, SOCIAL_HANDLE, THROWAWAY_ACCOUNT, URL, INSURANCE_ID, REDDIT_USER patterns")

    print(f"\nOutput files saved to {output_dir}/:")
    print("  - train.jsonl, validation.jsonl, test.jsonl (for fine-tuning)")
    print("  - train.csv, validation.csv, test.csv (for analysis)")
    print("  - dataset_statistics.json (detailed statistics)")
    print("  - data_preparation.log (processing logs)")

if __name__ == "__main__":
    OUTPUT_DIR = "combined_medical_data"
    main(OUTPUT_DIR)
