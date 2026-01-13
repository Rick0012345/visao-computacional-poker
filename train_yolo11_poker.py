#!/usr/bin/env python3
"""
YOLOv11 Poker Card Training Script
Treina um modelo YOLOv11 para detectar cartas de poker
"""

import os
import sys
import yaml
import torch
from pathlib import Path
import argparse
from datetime import datetime

try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Ultralytics não está instalado. Instale com: pip install ultralytics")
    sys.exit(1)

def create_project_structure():
    """Cria estrutura de diretórios para o projeto"""
    dirs = [
        'poker_yolo11',
        'poker_yolo11/models',
        'poker_yolo11/runs',
        'poker_yolo11/datasets',
        'poker_yolo11/results'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Estrutura de diretórios criada")

def create_data_yaml(dataset_path, output_path='poker_yolo11/datasets/poker_data.yaml'):
    """
    Cria arquivo data.yaml para treinamento
    
    Args:
        dataset_path: Caminho para o dataset
        output_path: Caminho para salvar o data.yaml
    """
    
    # Verificar se o dataset existe
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return None
    
    # Criar configuração do dataset
    data_config = {
        'train': str(dataset_path / 'train' / 'images'),
        'val': str(dataset_path / 'valid' / 'images'),
        'test': str(dataset_path / 'test' / 'images') if (dataset_path / 'test').exists() else None,
        'nc': 52,  # Número de classes (52 cartas do baralho)
        'names': [
            '10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
            '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
            '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
            'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
            'Qc', 'Qd', 'Qh', 'Qs'
        ]
    }
    
    # Salvar configuração
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Arquivo data.yaml criado: {output_path}")
    return str(output_path)

def train_model(data_yaml, model_size='n', epochs=100, imgsz=640, batch_size=16, patience=20):
    """
    Treina o modelo YOLOv11
    
    Args:
        data_yaml: Caminho para o arquivo data.yaml
        model_size: Tamanho do modelo ('n' para nano, 's' para small, 'm' para medium, 'l' para large, 'x' para extra large)
        epochs: Número de épocas
        imgsz: Tamanho das imagens
        batch_size: Tamanho do batch
        patience: Paciência para early stopping
    
    Returns:
        Path: Caminho para o melhor modelo treinado
    """
    
    print(f"🚀 Iniciando treinamento YOLOv11-{model_size}")
    print(f"📊 Dataset: {data_yaml}")
    print(f"⏱️ Épocas: {epochs}")
    print(f"📐 Tamanho da imagem: {imgsz}x{imgsz}")
    print(f"📦 Batch size: {batch_size}")
    
    # Selecionar dispositivo
    device = select_device('' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Dispositivo: {device}")
    
    # Criar modelo
    model_name = f'yolo11{model_size}.pt'
    print(f"📥 Carregando modelo base: {model_name}")
    
    try:
        model = YOLO(model_name)
        
        # Configurar hiperparâmetros
        model.overrides = {
            'conf': 0.25,  # Confidence threshold
            'iou': 0.45,   # NMS IoU threshold
            'agnostic_nms': False,
            'max_det': 300,  # Maximum detections per image
            'amp': True,  # Automatic Mixed Precision
        }
        
        # Treinar modelo
        print("🏋️ Iniciando treinamento...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            save_period=10,
            project='poker_yolo11',
            name=f'poker_detector_{model_size}',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,  # Initial learning rate
            lrf=0.01,   # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,    # Box loss gain
            cls=0.5,    # Classification loss gain
            dfl=1.5,    # Distribution Focal Loss gain
            # Augmentations
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )
        
        # Obter caminho do melhor modelo
        best_model_path = Path('poker_yolo11') / f'poker_detector_{model_size}' / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"✅ Treinamento concluído! Melhor modelo: {best_model_path}")
            return str(best_model_path)
        else:
            print("⚠️ Treinamento concluído, mas não foi possível encontrar o melhor modelo")
            return None
            
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        return None

def validate_model(model_path, data_yaml):
    """
    Valida o modelo treinado
    
    Args:
        model_path: Caminho para o modelo treinado
        data_yaml: Caminho para o arquivo data.yaml
    
    Returns:
        Dict: Métricas de validação
    """
    print(f"🔍 Validando modelo: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # Validar no conjunto de validação
        results = model.val(data=data_yaml)
        
        # Extrair métricas
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1_score': results.box.f1,
        }
        
        print("📊 Métricas de validação:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro durante a validação: {e}")
        return None

def export_model(model_path, format='onnx'):
    """
    Exporta o modelo para diferentes formatos
    
    Args:
        model_path: Caminho para o modelo treinado
        format: Formato de exportação ('onnx', 'torchscript', 'tflite', etc.)
    
    Returns:
        Path: Caminho para o modelo exportado
    """
    print(f"📤 Exportando modelo para {format.upper()}")
    
    try:
        model = YOLO(model_path)
        
        # Exportar modelo
        exported_model = model.export(format=format, optimize=True)
        
        print(f"✅ Modelo exportado: {exported_model}")
        return exported_model
        
    except Exception as e:
        print(f"❌ Erro durante a exportação: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Treina um modelo YOLOv11 para detecção de cartas de poker')
    parser.add_argument('--dataset', type=str, default='cards_templates/IA/dataset',
                       help='Caminho para o dataset YOLOv11')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Tamanho do modelo (n=slower/better, x=faster/worse)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas para treinamento')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Tamanho das imagens')
    parser.add_argument('--batch', type=int, default=16,
                       help='Tamanho do batch')
    parser.add_argument('--patience', type=int, default=20,
                       help='Paciência para early stopping')
    parser.add_argument('--validate', action='store_true',
                       help='Validar modelo após treinamento')
    parser.add_argument('--export', type=str, choices=['onnx', 'torchscript', 'tflite'],
                       help='Exportar modelo para formato especificado')
    
    args = parser.parse_args()
    
    print("🃏 YOLOv11 Poker Card Training")
    print("=" * 50)
    
    # Criar estrutura do projeto
    create_project_structure()
    
    # Criar arquivo data.yaml
    data_yaml = create_data_yaml(args.dataset)
    if not data_yaml:
        print("❌ Não foi possível criar o arquivo data.yaml")
        return
    
    # Treinar modelo
    best_model = train_model(
        data_yaml=data_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        patience=args.patience
    )
    
    if not best_model:
        print("❌ Treinamento falhou")
        return
    
    # Validar modelo se solicitado
    if args.validate:
        validate_model(best_model, data_yaml)
    
    # Exportar modelo se solicitado
    if args.export:
        export_model(best_model, args.export)
    
    print("\n🎉 Processo concluído!")
    print(f"📁 Modelo salvo em: {best_model}")
    print("💡 Use este modelo com o YOLOPokerDetector")

if __name__ == "__main__":
    main()