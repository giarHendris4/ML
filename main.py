import argparse
import yaml
import sys
from src.data_loader import DataLoader
from src.trainer import ModelTrainer
from src.predictor import Predictor

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' tidak ditemukan.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Config file '{config_path}' tidak valid YAML. Detail: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='ML Simple Trainer - Train and predict with machine learning models'
    )
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='Mode operasi: train untuk melatih model, predict untuk prediksi')
    parser.add_argument('--data', required=True,
                        help='Path ke file CSV data (training atau prediksi)')
    parser.add_argument('--config', default='config.yaml',
                        help='Path ke file konfigurasi YAML (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        if args.mode == 'train':
            print(f"Memuat data dari: {args.data}")
            X_train, X_test, y_train, y_test, problem_type = DataLoader.load_and_split(
                file_path=args.data,
                target_column=config.get('target_column'),
                test_size=config.get('test_size', 0.2),
                random_state=config.get('random_state', 42)
            )
            
            print(f"Problem type terdeteksi: {problem_type}")
            print(f"Training set size: {len(X_train)} samples")
            print(f"Test set size: {len(X_test)} samples")
            
            print("Melatih model...")
            trainer = ModelTrainer(
                model_type=config.get('model_type', 'random_forest'),
                problem_type=problem_type
            )
            trainer.train(X_train, y_train)
            
            metrics = trainer.evaluate(X_test, y_test)
            print(f"Hasil evaluasi: {metrics}")
            
            model_path = trainer.save()
            print(f"Model disimpan ke: {model_path}")
            print("Training selesai.")
            
        elif args.mode == 'predict':
            print(f"Memuat model dari: models/model.pkl")
            model = Predictor.load_model()
            
            print(f"Memuat data prediksi dari: {args.data}")
            predictions = Predictor.predict(model, args.data)
            
            print("Hasil prediksi:")
            for i, pred in enumerate(predictions):
                print(f"  Sample {i+1}: {pred}")
            
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error tidak terduga: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
