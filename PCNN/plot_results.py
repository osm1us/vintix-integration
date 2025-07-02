import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_results(log_file_path, window_size=50):
    """
    Читает лог-файл обучения и строит график награды.

    :param log_file_path: Путь к файлу training_log.csv.
    :param window_size: Размер окна для скользящего среднего.
    """
    if not os.path.exists(log_file_path):
        print(f"Ошибка: Файл не найден по пути: {log_file_path}")
        return

    try:
        df = pd.read_csv(log_file_path)
    except pd.errors.EmptyDataError:
        print(f"Ошибка: Лог-файл '{log_file_path}' пуст. Возможно, обучение еще не началось.")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    
    if 'reward' not in df.columns or 'episode' not in df.columns:
        print("Ошибка: В лог-файле отсутствуют необходимые колонки 'episode' или 'reward'.")
        return

    # Рассчитываем скользящее среднее для сглаживания графика
    df['reward_smoothed'] = df['reward'].rolling(window=window_size, min_periods=1).mean()

    # Строим график
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df['episode'], df['reward'], color='lightblue', alpha=0.5, label='Награда за эпизод')
    ax.plot(df['episode'], df['reward_smoothed'], color='dodgerblue', linewidth=2, label=f'Скользящее среднее (окно={window_size})')
    
    ax.set_title(f'Динамика обучения: {os.path.basename(log_file_path)}', fontsize=16)
    ax.set_xlabel('Эпизод', fontsize=12)
    ax.set_ylabel('Общая награда', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    print("График готов. Показываю окно...")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Визуализация результатов обучения RL-агента.")
    parser.add_argument(
        'log_path', 
        type=str, 
        help="Путь к лог-файлу (training_log.csv) или к папке с моделями, содержащей этот файл."
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=50,
        help="Размер окна для скользящего среднего (по умолчанию: 50)."
    )
    
    args = parser.parse_args()

    target_path = args.log_path
    if os.path.isdir(target_path):
        log_file = os.path.join(target_path, 'training_log.csv')
        if not os.path.exists(log_file):
            print(f"Ошибка: В папке '{target_path}' не найден файл 'training_log.csv'.")
            exit()
        target_path = log_file

    plot_training_results(target_path, args.window) 