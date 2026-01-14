#!/usr/bin/env python3
"""
Debug do erro de treinamento YOLO
"""

import sys
import os
from pathlib import Path

def debug_yolo_train():
    """Debug do erro de treinamento"""
    
    print("🔍 Debug do Erro de Treinamento YOLO")
    print("=" * 50)
    
    # 1. Verificar versão do Ultralytics
    try:
        import ultralytics
        print(f"✅ Ultralytics versão: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ Ultralytics não instalado: {e}")
        return False
    
    # 2. Verificar importação do YOLO
    try:
        from ultralytics import YOLO
        print("✅ YOLO importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar YOLO: {e}")
        return False
    
    # 3. Verificar modelo base
    print("\n📦 Verificando modelos base...")
    
    model_sizes = ['n', 's', 'm', 'l', 'x']
    
    for size in model_sizes:
        model_name = f'yolo11{size}.pt'
        try:
            print(f"\n🎯 Testando YOLOv11-{size}...")
            model = YOLO(model_name)
            print(f"✅ YOLOv11-{size} carregado com sucesso")
            
            # Mostrar informações do modelo
            print(f"   Modelo: {model_name}")
            print(f"   Tasks: {list(model.model.names.values())[:5]}...")  # Primeiras 5 classes
            
            # Limpar memória
            del model
            break  # Se um funcionar, os outros também devem
            
        except Exception as e:
            print(f"❌ Erro ao carregar YOLOv11-{size}: {e}")
            print(f"   Detalhes: {type(e).__name__}: {str(e)[:100]}...")
    
    # 4. Verificar dataset
    print("\n📁 Verificando dataset...")
    dataset_path = Path("cards_templates/IA/dataset")
    data_yaml = dataset_path / "data.yaml"
    
    if dataset_path.exists():
        print(f"✅ Pasta dataset encontrada: {dataset_path}")
        
        if data_yaml.exists():
            print(f"✅ Arquivo data.yaml encontrado: {data_yaml}")
            
            # Ler conteúdo do data.yaml
            try:
                import yaml
                with open(data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)
                
                print(f"   Train: {data_config.get('train', 'N/A')}")
                print(f"   Val: {data_config.get('val', 'N/A')}")
                print(f"   Classes: {data_config.get('nc', 'N/A')}")
                print(f"   Names: {data_config.get('names', [])[:5]}...")  # Primeiras 5 classes
                
            except Exception as e:
                print(f"❌ Erro ao ler data.yaml: {e}")
        else:
            print(f"❌ Arquivo data.yaml não encontrado: {data_yaml}")
    else:
        print(f"❌ Pasta dataset não encontrada: {dataset_path}")
    
    # 5. Teste simples de treinamento
    print("\n🧪 Testando treinamento simples...")
    
    try:
        from ultralytics import YOLO
        
        # Usar modelo nano para teste rápido
        model = YOLO('yolo11n.pt')
        print("✅ Modelo base carregado")
        
        # Testar treinamento com 1 época apenas
        if data_yaml.exists():
            print("🎯 Iniciando treinamento de teste (1 época)...")
            
            results = model.train(
                data=str(data_yaml),
                epochs=1,
                imgsz=640,
                batch=8,
                project='debug_yolo',
                name='test_train',
                exist_ok=True,
                patience=1,
                save=False,  # Não salvar para economizar espaço
                pretrained=True,
            )
            
            print("✅ Treinamento de teste concluído!")
            print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            
        else:
            print("⚠️ Dataset não disponível para teste")
            
    except Exception as e:
        print(f"❌ Erro no treinamento de teste: {e}")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Detalhes: {str(e)}")
        
        # Tentar obter mais informações sobre o erro
        import traceback
        print("\n📋 Stack trace completo:")
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🔍 Debug concluído!")
    
    return True

if __name__ == "__main__":
    debug_yolo_train()