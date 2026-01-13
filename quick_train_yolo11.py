#!/usr/bin/env python3
"""
Quick YOLOv11 Poker Training
Treinamento rápido do modelo YOLOv11 para cartas de poker
"""

import sys
import os
from pathlib import Path
import time

def quick_train():
    """Treinamento rápido do YOLOv11"""
    
    print("🃏 YOLOv11 Poker - Quick Training")
    print("=" * 40)
    
    # Verificar dependências
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics disponível")
    except ImportError:
        print("❌ Ultralytics não instalado")
        print("   Execute: pip install ultralytics")
        return False
    
    # Verificar dataset
    dataset_path = Path("cards_templates/IA/dataset")
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        print("❌ Dataset não encontrado")
        print(f"   Verifique: {data_yaml}")
        return False
    
    print(f"✅ Dataset encontrado: {dataset_path}")
    
    # Configurações rápidas
    model_size = 'n'  # Nano - mais rápido
    epochs = 50       # Menos épocas para treino rápido
    imgsz = 640      # Tamanho padrão
    batch = 16       # Batch pequeno
    
    print(f"📊 Configurações:")
    print(f"   Modelo: YOLOv11-{model_size}")
    print(f"   Épocas: {epochs}")
    print(f"   Batch: {batch}")
    print(f"   Imagem: {imgsz}x{imgsz}")
    
    # Confirmar treinamento
    response = input("\n🚀 Iniciar treinamento? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Treinamento cancelado")
        return False
    
    try:
        print("\n🏋️ Iniciando treinamento...")
        
        # Carregar modelo
        model = YOLO(f'yolo11{model_size}.pt')
        print(f"✅ Modelo YOLOv11-{model_size} carregado")
        
        # Treinar
        print("🎯 Treinando...")
        start_time = time.time()
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project='poker_yolo11',
            name=f'quick_train_{model_size}',
            exist_ok=True,
            patience=10,  # Early stopping rápido
            save=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            # Augmentações básicas
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
        )
        
        training_time = time.time() - start_time
        
        # Resultados
        print(f"\n✅ Treinamento concluído em {training_time/60:.1f} minutos!")
        
        # Verificar melhor modelo
        best_model = Path('poker_yolo11') / f'quick_train_{model_size}' / 'weights' / 'best.pt'
        
        if best_model.exists():
            print(f"🎯 Melhor modelo salvo em: {best_model}")
            
            # Testar modelo
            print("\n🧪 Testando modelo...")
            test_model = YOLO(str(best_model))
            
            # Validar rapidamente
            val_results = test_model.val(data=str(data_yaml))
            
            print(f"📊 Resultados de validação:")
            print(f"   mAP50: {val_results.box.map50:.3f}")
            print(f"   mAP50-95: {val_results.box.map:.3f}")
            
            return True
        else:
            print("❌ Modelo não encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        return False

def main():
    """Função principal"""
    
    # Verificar se devemos usar configurações personalizadas
    if len(sys.argv) > 1:
        # Modo personalizado
        try:
            from train_yolo11_poker import main as advanced_train
            print("🎯 Usando modo avançado...")
            advanced_train()
        except ImportError:
            print("❌ Modo avançado não disponível")
            print("   Usando modo rápido...")
            quick_train()
    else:
        # Modo rápido padrão
        quick_train()

if __name__ == "__main__":
    main()