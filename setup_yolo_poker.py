#!/usr/bin/env python3
"""
Setup script para YOLOv11 Poker Detection
Instala todas as dependências necessárias
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_command(command, description=None):
    """Executa comando e retorna sucesso/erro"""
    if description:
        print(f"📦 {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Sucesso: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro: {description or command}")
        print(f"   Saída: {e.stderr}")
        return False

def install_package(package, description=None):
    """Instala pacote Python"""
    cmd = f"{sys.executable} -m pip install {package}"
    return run_command(cmd, description or f"Instalando {package}")

def check_gpu_support():
    """Verifica suporte a GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU disponível: {torch.cuda.get_device_name(0)}")
            print(f"📊 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("💻 GPU não disponível, usando CPU")
            return False
    except ImportError:
        print("❌ PyTorch não instalado")
        return False

def main():
    print("🃏 YOLOv11 Poker Detection - Setup")
    print("=" * 50)
    
    # Verificar Python versão
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ é necessário")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Atualizar pip
    print("\n📈 Atualizando pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Atualizando pip")
    
    # Instalar PyTorch (CPU ou GPU)
    print("\n🔥 Instalando PyTorch...")
    
    # Tentar instalar PyTorch com GPU support primeiro
    gpu_success = install_package("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", 
                                 "PyTorch com GPU support")
    
    if not gpu_success:
        # Fallback para CPU
        install_package("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", 
                       "PyTorch CPU (fallback)")
    
    # Instalar Ultralytics (YOLOv8/v11)
    print("\n🎯 Instalando Ultralytics (YOLO)...")
    install_package("ultralytics", "Ultralytics YOLO")
    
    # Instalar OpenCV
    print("\n📸 Instalando OpenCV...")
    install_package("opencv-python", "OpenCV")
    
    # Instalar outras dependências
    print("\n📦 Instalando outras dependências...")
    
    dependencies = [
        ("numpy", "NumPy"),
        ("pillow", "Pillow (Image Processing)"),
        ("pyautogui", "PyAutoGUI (Screen Capture)"),
        ("mss", "MSS (Screen Capture)"),
        ("treys", "Treys (Poker Evaluation)"),
        ("matplotlib", "Matplotlib (Visualization)"),
        ("tqdm", "TQDM (Progress Bars)"),
        ("pyyaml", "PyYAML (Config Files)"),
    ]
    
    for package, description in dependencies:
        install_package(package, description)
    
    # Verificar GPU support
    print("\n🔍 Verificando suporte a GPU...")
    has_gpu = check_gpu_support()
    
    # Testar instalações
    print("\n🧪 Testando instalações...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics"),
    ]
    
    if has_gpu:
        test_imports.append(("torch.cuda", "CUDA PyTorch"))
    
    all_success = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - Falhou: {e}")
            all_success = False
    
    # Criar diretórios necessários
    print("\n📁 Criando estrutura de diretórios...")
    
    directories = [
        "poker_yolo11",
        "poker_yolo11/models",
        "poker_yolo11/datasets",
        "poker_yolo11/runs",
        "poker_yolo11/results",
        "lixeira",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Criado: {directory}")
    
    # Verificar dataset
    print("\n📊 Verificando dataset...")
    dataset_path = Path("cards_templates/IA/dataset")
    
    if dataset_path.exists():
        train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
        valid_images = list((dataset_path / "valid" / "images").glob("*.jpg"))
        
        print(f"✅ Dataset encontrado!")
        print(f"   📸 Imagens de treino: {len(train_images)}")
        print(f"   🔍 Imagens de validação: {len(valid_images)}")
        
        # Verificar data.yaml
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            print(f"✅ Arquivo data.yaml encontrado")
        else:
            print("⚠️ Arquivo data.yaml não encontrado")
    else:
        print("⚠️ Dataset não encontrado em: cards_templates/IA/dataset")
        print("   💡 Baixe o dataset YOLOv11 do Roboflow e extraia neste diretório")
    
    # Criar arquivo de configuração
    print("\n⚙️ Criando arquivo de configuração...")
    
    config_content = f"""# YOLOv11 Poker Configuration
# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Model settings
MODEL_SIZE: n  # n, s, m, l, x
CONFIDENCE: 0.5
IOU_THRESHOLD: 0.45

# Training settings
EPOCHS: 100
BATCH_SIZE: 16
IMAGE_SIZE: 640
PATIENCE: 20

# Detection settings
USE_YOLO: true
USE_TEMPLATE_FALLBACK: true
DETECTION_SMOOTHING: true
BUFFER_SIZE: 3

# Paths
DATASET_PATH: cards_templates/IA/dataset
DEFAULT_MODEL: poker_yolo11/poker_detector_n/weights/best.pt

# GPU Settings
GPU_AVAILABLE: {has_gpu}
DEVICE: {'cuda' if has_gpu else 'cpu'}
"""
    
    config_path = Path("poker_yolo11/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Configuração salva em: {config_path}")
    
    # Resultado final
    print("\n" + "=" * 50)
    if all_success:
        print("🎉 Setup concluído com sucesso!")
        print("\n📖 Próximos passos:")
        print("1. Treine o modelo: python train_yolo11_poker.py")
        print("2. Teste o detector: python yolo_poker_analyzer.py")
        print("3. Use na interface: O poker_gui_compact.py usará automaticamente")
        
        if has_gpu:
            print("\n🚀 GPU detectada! O treinamento será acelerado.")
        else:
            print("\n💻 Usando CPU. O treinamento será mais lento mas funcionará.")
            
    else:
        print("⚠️ Setup concluído com avisos!")
        print("   Algumas dependências podem estar faltando.")
        print("   Tente rodar: pip install -r requirements.txt")
    
    print(f"\n📁 Arquivos criados:")
    print(f"   📂 poker_yolo11/ - Diretório do projeto")
    print(f"   📄 poker_yolo11/config.yaml - Configuração")
    print(f"   📂 lixeira/ - Para screenshots")
    
    print("\n💡 Dica: Use 'python train_yolo11_poker.py --help' para ver opções de treinamento")

if __name__ == "__main__":
    main()
