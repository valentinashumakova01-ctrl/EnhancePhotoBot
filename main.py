# main.py
import yaml
from training.train import Trainer

def main():
    # Загрузка конфигурации
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Запуск тренировки
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
