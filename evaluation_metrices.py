import json
import sys
sys.path.append('/content/drive/My Drive/mistral_mental_medical_chatbot')
import logging
import os
from datetime import datetime
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import nltk
from mistral_chatbot import MistralChatbot, SafetyGuardrails, QueryRoutingClassifier
from typing import List, Dict, Tuple
from huggingface_hub import login
import time
import multiprocessing as mp
import torch
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Setup logging function
def setup_evaluation_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "/content/drive/My Drive/mistral_mental_medical_chatbot/logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f'{log_dir}/evaluation_{timestamp}.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

# Setup logging at the start
logger = setup_evaluation_logging()

# Login to Hugging Face (optional if models are local)
try:
    login(token="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")
    logger.info("Hugging Face login successful")
except Exception as e:
    logger.warning(f"Hugging Face login failed: {e}. Proceeding with local models.")

def preprocess_text(text):
    """Preprocess text for evaluation metrics"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?-]', '', text)
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    return text

def calculate_bleu(generated, reference):
    """BLEU calculation with multiple n-gram weights"""
    generated_tokens = generated.split()
    reference_tokens = [reference.split()]

    if len(generated_tokens) == 0 or len(reference_tokens[0]) == 0:
        return 0.0

    smoothing = SmoothingFunction().method1
    weights_1 = (1.0, 0, 0, 0)
    weights_2 = (0.5, 0.5, 0, 0)
    weights_4 = (0.25, 0.25, 0.25, 0.25)

    try:
        bleu_1 = sentence_bleu(reference_tokens, generated_tokens, weights=weights_1, smoothing_function=smoothing)
        bleu_2 = sentence_bleu(reference_tokens, generated_tokens, weights=weights_2, smoothing_function=smoothing)
        bleu_4 = sentence_bleu(reference_tokens, generated_tokens, weights=weights_4, smoothing_function=smoothing)
        bleu = 0.5 * bleu_1 + 0.3 * bleu_2 + 0.2 * bleu_4
        return bleu
    except:
        return 0.0

def calculate_rouge_score(generated, reference):
    """Calculate ROUGE-1 score"""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores['rouge1'].fmeasure

def calculate_medical_metrics(generated, reference):
    """Precision, recall, and F1 for medical/mental health content"""
    medical_keywords = {
        'diagnosis', 'treatment', 'symptoms', 'testing', 'disease', 'condition',
        'therapy', 'medication', 'syndrome', 'disorder', 'patient', 'clinical',
        'medical', 'health', 'doctor', 'physician', 'hospital', 'pain',
        'infection', 'chronic', 'acute', 'prognosis', 'pathology',
        'anxiety', 'depression', 'stress', 'counseling', 'psychotherapy',
        'psychiatrist', 'psychologist', 'mental', 'emotional', 'psychological',
        'cognitive', 'behavioral', 'mindfulness', 'coping', 'support',
        'wellness', 'self-care', 'resilience', 'trauma', 'ptsd',
        'bipolar', 'schizophrenia', 'ocd', 'adhd', 'autism',
        'cbt', 'dbt', 'emdr', 'intervention', 'prevention', 'recovery',
        'rehabilitation', 'therapeutic', 'holistic', 'integrated'
    }

    gen_words = set(preprocess_text(generated).split())
    ref_words = set(preprocess_text(reference).split())
    gen_medical = gen_words & medical_keywords
    ref_medical = ref_words & medical_keywords
    common_medical = gen_medical & ref_medical
    common_words = gen_words & ref_words

    medical_weight = 2.0
    general_weight = 1.0

    true_positives = len(common_words) * general_weight + len(common_medical) * medical_weight
    predicted_positives = len(gen_words) * general_weight + len(gen_medical) * medical_weight
    actual_positives = len(ref_words) * general_weight + len(ref_medical) * medical_weight

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

class OpenAIEmpathyScorer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def score_empathy(self, response: str, domain: str, question: str = "") -> float:
        """Empathy scoring with context awareness"""
        if domain != 'mental_health':
            return None

        try:
            prompt = f"""
Evaluate the empathy level of the following mental health response on a scale from 0.0 to 1.0.

Original Question: "{question}"
Response to evaluate: "{response}"

Scoring criteria (0.0 to 1.0):
- 0.0-0.2: Harmful, dismissive, or completely inappropriate
- 0.2-0.4: Lacking empathy, too clinical or cold
- 0.4-0.6: Neutral, informational but shows some understanding
- 0.6-0.8: Good empathy, supportive and understanding
- 0.8-1.0: Excellent empathy, highly supportive, validates feelings

Consider:
1. Acknowledgment of emotions and feelings
2. Supportive and non-judgmental language
3. Validation of the person's experience
4. Appropriate suggestions for help/resources
5. Warmth and understanding in tone
6. Avoidance of dismissive phrases
7. Crisis resource provision when needed

Respond with only a decimal number between 0.0 and 1.0.
"""

            response_obj = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in psychological assessment specializing in empathy evaluation for mental health responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )

            score_text = response_obj.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Error getting empathy score from OpenAI: {e}")
            return self._fallback_empathy_score(response, question)

    def _fallback_empathy_score(self, response: str, question: str = "") -> float:
        """Fallback empathy scoring"""
        empathy_indicators = {
            'high_positive': [
                'i understand how you feel', 'that sounds really difficult', 'you\'re not alone in this',
                'it takes courage to reach out', 'your feelings are valid', 'i hear you',
                'that must be overwhelming', 'it\'s okay to feel this way', 'you deserve support'
            ],
            'moderate_positive': [
                'understand', 'difficult', 'challenging', 'support', 'help available',
                'normal to feel', 'common experience', 'many people feel', 'you matter'
            ],
            'negative': [
                'just get over it', 'stop worrying', 'it\'s not a big deal', 'you shouldn\'t feel',
                'snap out of it', 'just think positive', 'others have it worse'
            ],
            'crisis_positive': [
                '988', 'crisis text line', 'national suicide prevention', 'emergency services',
                'crisis hotline', 'immediate help', 'call 911', 'emergency room'
            ]
        }

        response_lower = response.lower()
        score = 0.3

        for phrase in empathy_indicators['high_positive']:
            if phrase in response_lower:
                score += 0.2
        for phrase in empathy_indicators['moderate_positive']:
            if phrase in response_lower:
                score += 0.1
        for phrase in empathy_indicators['negative']:
            if phrase in response_lower:
                score -= 0.3
        for phrase in empathy_indicators['crisis_positive']:
            if phrase in response_lower:
                score += 0.15

        if len(response.split()) > 50:
            score += 0.05
        if len(response.split()) > 100:
            score += 0.05

        return max(0.0, min(1.0, score))

def evaluate_single_sample(args):
    try:
        i, question, reference_answer, source, model_path, routing_classifier_path, openai_api_key = args
        chatbot = MistralChatbot(model_path, routing_classifier_path)
        chatbot.load_models()
        empathy_scorer = OpenAIEmpathyScorer(openai_api_key)

        response_dict = chatbot.generate_response(question)
        generated_response = response_dict['response']
        domain = response_dict['domain']

        gen_processed = preprocess_text(generated_response)
        ref_processed = preprocess_text(reference_answer)

        empathy_score = empathy_scorer.score_empathy(generated_response, domain, question)

        try:
            bleu_score = calculate_bleu(gen_processed, ref_processed)
        except Exception as e:
            logger.warning(f"Error computing BLEU for sample {i}: {e}")
            bleu_score = 0.0

        try:
            rouge_score = calculate_rouge_score(gen_processed, ref_processed)
        except Exception as e:
            logger.warning(f"Error computing ROUGE for sample {i}: {e}")
            rouge_score = 0.0

        try:
            precision, recall, f1 = calculate_medical_metrics(generated_response, reference_answer)
        except Exception as e:
            logger.warning(f"Error computing precision/recall/F1 for sample {i}: {e}")
            precision, recall, f1 = 0.0, 0.0, 0.0

        if hasattr(chatbot, 'model') and hasattr(chatbot.model, 'cpu'):
            chatbot.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'index': i,
            'empathy_score': empathy_score,
            'bleu_score': bleu_score,
            'rouge_score': rouge_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'question': question,
            'generated_response': generated_response,
            'reference_answer': reference_answer,
            'domain': domain,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error evaluating sample {i}: {e}")
        return {
            'index': i,
            'success': False,
            'error': str(e)
        }

class ModelEvaluator:
    def __init__(self, model_path: str, routing_classifier_path: str, test_data_path: str, openai_api_key: str):
        self.model_path = model_path
        self.routing_classifier_path = routing_classifier_path
        self.test_data_path = test_data_path
        self.openai_api_key = openai_api_key
        self.empathy_scorer = OpenAIEmpathyScorer(openai_api_key)
        self.test_data = self.load_test_data()
        self.preprocessed_data = [(item['text'].split('Answer:')[0].replace('Question:', '').strip(),
                                 item['text'].split('Answer:')[1].strip(),
                                 item['source']) for item in self.test_data]
        self.responses_for_manual_review = []
        self.start_time = None
        self.total_samples = 0

    def load_test_data(self) -> List[Dict]:
        try:
            data = []
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON: {e}")
            logger.info(f"Loaded {len(data)} test samples from {self.test_data_path}")
            self.total_samples = min(len(data), 1000)
            return data
        except FileNotFoundError:
            logger.error(f"Test data file not found: {self.test_data_path}")
            return []

    def evaluate_sequential(self, max_samples: int = 100, manual_review_samples: int = 100):
        if not self.test_data:
            logger.error("No test data available. Aborting evaluation.")
            print("Error: No test data available. Please check test.jsonl.")
            return

        try:
            chatbot = MistralChatbot(self.model_path, self.routing_classifier_path)
            chatbot.load_models()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

        self.start_time = time.time()
        logger.info("Starting sequential evaluation...")
        print(f"Evaluation started at {datetime.now().strftime('%H:%M:%S')}")

        empathy_scores = []
        bleu_scores = []
        rouge_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        domain_counts = {'mental_health': 0, 'medical': 0, 'general': 0}

        test_samples = self.preprocessed_data[:min(len(self.test_data), max_samples)]

        for i, (question, reference_answer, source) in enumerate(test_samples):
            try:
                response_dict = chatbot.generate_response(question)
                generated_response = response_dict['response']
                domain = response_dict['domain']

                if domain in domain_counts:
                    domain_counts[domain] += 1
                else:
                    domain_counts['general'] += 1

                gen_processed = preprocess_text(generated_response)
                ref_processed = preprocess_text(reference_answer)

                empathy_score = self.empathy_scorer.score_empathy(generated_response, domain, question)
                if empathy_score is not None:
                    empathy_scores.append(empathy_score)

                try:
                    bleu_score = calculate_bleu(gen_processed, ref_processed)
                    bleu_scores.append(bleu_score)
                except Exception as e:
                    logger.warning(f"Error computing BLEU for sample {i}: {e}")
                    bleu_scores.append(0.0)

                try:
                    rouge_score = calculate_rouge_score(gen_processed, ref_processed)
                    rouge_scores.append(rouge_score)
                except Exception as e:
                    logger.warning(f"Error computing ROUGE for sample {i}: {e}")
                    rouge_scores.append(0.0)

                try:
                    precision, recall, f1 = calculate_medical_metrics(generated_response, reference_answer)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                except Exception as e:
                    logger.warning(f"Error computing precision/recall/F1 for sample {i}: {e}")
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
                    f1_scores.append(0.0)

                if i < manual_review_samples:
                    self.responses_for_manual_review.append({
                        'question': question,
                        'generated_response': generated_response,
                        'reference_answer': reference_answer,
                        'domain': domain,
                        'empathy_score': empathy_score,
                        'bleu': bleu_scores[-1],
                        'rouge': rouge_scores[-1],
                        'precision': precision_scores[-1],
                        'recall': recall_scores[-1],
                        'f1': f1_scores[-1]
                    })

                elapsed_time = time.time() - self.start_time
                avg_time_per_sample = elapsed_time / (i + 1)
                estimated_remaining = (max_samples - i - 1) * avg_time_per_sample
                percent_complete = ((i + 1) / max_samples) * 100

                sys.stdout.write(f"\rProgress: {i+1}/{max_samples} ({percent_complete:.1f}%), "
                               f"Elapsed: {elapsed_time:.1f}s, Est. Remaining: {estimated_remaining:.1f}s, "
                               f"Mental Health Qs: {len(empathy_scores)}")
                sys.stdout.flush()

                if torch.cuda.is_available() and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Error evaluating sample {i+1}: {e}")
                bleu_scores.append(0.0)
                rouge_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                f1_scores.append(0.0)

        end_time = time.time()
        total_time = end_time - self.start_time
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")

        results = {
            'empathy': {
                'mean': float(np.mean(empathy_scores)) if empathy_scores else 0.0,
                'std': float(np.std(empathy_scores)) if empathy_scores else 0.0,
                'count': len(empathy_scores),
                'note': 'Only calculated for mental_health domain questions'
            },
            'bleu': {
                'mean': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
                'std': float(np.std(bleu_scores)) if bleu_scores else 0.0,
                'count': len(bleu_scores)
            },
            'rouge': {
                'mean': float(np.mean(rouge_scores)) if rouge_scores else 0.0,
                'std': float(np.std(rouge_scores)) if rouge_scores else 0.0,
                'count': len(rouge_scores)
            },
            'precision': {
                'mean': float(np.mean(precision_scores)) if precision_scores else 0.0,
                'std': float(np.std(precision_scores)) if precision_scores else 0.0,
                'count': len(precision_scores)
            },
            'recall': {
                'mean': float(np.mean(recall_scores)) if recall_scores else 0.0,
                'std': float(np.std(recall_scores)) if recall_scores else 0.0,
                'count': len(recall_scores)
            },
            'f1': {
                'mean': float(np.mean(f1_scores)) if f1_scores else 0.0,
                'std': float(np.std(f1_scores)) if f1_scores else 0.0,
                'count': len(f1_scores)
            },
            'domain_distribution': domain_counts,
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/content/drive/My Drive/mistral_mental_medical_chatbot"
        results_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        manual_review_path = os.path.join(output_dir, f'manual_review_samples_{timestamp}.json')
        with open(manual_review_path, 'w') as f:
            json.dump(self.responses_for_manual_review, f, indent=4)

        self._print_results(results, total_time, results_path, manual_review_path)

        return results

    def _print_results(self, results, total_time, results_path, manual_review_path):
        logger.info("Evaluation Results:")
        logger.info(f"Domain Distribution: {results['domain_distribution']}")
        logger.info(f"Empathy Score (Mental Health Only) - Mean: {results['empathy']['mean']:.4f}, Std: {results['empathy']['std']:.4f}, Count: {results['empathy']['count']}")
        logger.info(f"BLEU Score - Mean: {results['bleu']['mean']:.4f}, Std: {results['bleu']['std']:.4f}")
        logger.info(f"ROUGE Score - Mean: {results['rouge']['mean']:.4f}, Std: {results['rouge']['std']:.4f}")
        logger.info(f"Precision - Mean: {results['precision']['mean']:.4f}, Std: {results['precision']['std']:.4f}")
        logger.info(f"Recall - Mean: {results['recall']['mean']:.4f}, Std: {results['recall']['std']:.4f}")
        logger.info(f"F1 Score - Mean: {results['f1']['mean']:.4f}, Std: {results['f1']['std']:.4f}")
        logger.info(f"Total Time: {total_time:.2f} seconds")

        print(f"\n=== Evaluation Results ===")
        print(f"Domain Distribution: {results['domain_distribution']}")
        print(f"📊 Empathy Score (Mental Health Only) - Mean: {results['empathy']['mean']:.4f}, Std: {results['empathy']['std']:.4f} (n={results['empathy']['count']})")
        print(f"📝 BLEU Score - Mean: {results['bleu']['mean']:.4f}, Std: {results['bleu']['std']:.4f}")
        print(f"🔍 ROUGE Score - Mean: {results['rouge']['mean']:.4f}, Std: {results['rouge']['std']:.4f}")
        print(f"📈 Precision - Mean: {results['precision']['mean']:.4f}, Std: {results['precision']['std']:.4f}")
        print(f"📈 Recall - Mean: {results['recall']['mean']:.4f}, Std: {results['recall']['std']:.4f}")
        print(f"📈 F1 Score - Mean: {results['f1']['mean']:.4f}, Std: {results['f1']['std']:.4f}")
        print(f"🏁 Total Time: {total_time:.2f} seconds")
        print(f"📝 Results saved to: {results_path}")
        print(f"🔍 Manual Review Samples saved to: {manual_review_path}")
        print("\nFor ethical review, please check the manual_review_samples JSON file and assess responses for harmful advice.")

def main():
    openai_api_key = input("Enter your OpenAI API key: ").strip()
    if not openai_api_key:
        print("Error: OpenAI API key is required for empathy scoring.")
        return

    output_dir = "/content/drive/My Drive/mistral_mental_medical_chatbot"
    try:
        evaluator = ModelEvaluator(
            model_path=f"{output_dir}/mistral_model",
            routing_classifier_path=f"{output_dir}/routing_classifier",
            test_data_path=f"{output_dir}/test.jsonl",
            openai_api_key=openai_api_key
        )

        print("Choose evaluation method:")
        print("1. Sequential (recommended for GPU models)")
        print("2. Parallel (faster but may have issues with GPU)")

        choice = input("Enter choice (1 or 2, default=1): ").strip()

        if choice == "2":
            num_processes = int(input("Enter number of processes (default=2): ") or "2")
            results = evaluator.evaluate_parallel(max_samples=100, manual_review_samples=100, num_processes=num_processes)
        else:
            results = evaluator.evaluate_sequential(max_samples=300, manual_review_samples=100)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
