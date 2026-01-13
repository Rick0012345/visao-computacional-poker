#!/usr/bin/env python3
"""
Teste rápido da integração YOLO com Poker Analyzer
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Adicionar diretório atual ao path
sys.path.append(str(Path(__file__).parent))

def test_yolo_integration():
    """Testa a integração YOLO com o poker analyzer"""
    
    print("🃏 Testando Integração YOLO com Poker Analyzer")
    print("=" * 60)
    
    # Testar importações
    print("\n📦 Testando importações...")
    
    try:
        from poker_analyzer import PokerAnalyzer, YOLO_AVAILABLE
        print("✅ PokerAnalyzer importado com sucesso")
        print(f"   YOLO disponível: {YOLO_AVAILABLE}")
    except ImportError as e:
        print(f"❌ Erro ao importar PokerAnalyzer: {e}")
        return False
    
    try:
        from yolo_poker_analyzer import YOLOPokerAnalyzer
        print("✅ YOLOPokerAnalyzer importado com sucesso")
    except ImportError as e:
        print(f"⚠️ YOLOPokerAnalyzer não disponível: {e}")
    
    # Testar criação do analisador
    print("\n🔧 Testando criação do analisador...")
    
    try:
        # Criar analisador com YOLO habilitado
        analyzer = PokerAnalyzer(use_yolo=True, yolo_model_path=None)
        print("✅ PokerAnalyzer criado com sucesso")
        print(f"   YOLO habilitado: {analyzer.use_yolo}")
        
        if analyzer.yolo_analyzer:
            print("✅ YOLO Analyzer inicializado")
        else:
            print("ℹ️ YOLO Analyzer não disponível (modelo não encontrado)")
            
    except Exception as e:
        print(f"❌ Erro ao criar PokerAnalyzer: {e}")
        # Tentar sem YOLO
        try:
            analyzer = PokerAnalyzer(use_yolo=False)
            print("✅ PokerAnalyzer criado sem YOLO")
        except Exception as e2:
            print(f"❌ Erro ao criar PokerAnalyzer sem YOLO: {e2}")
            return False
    
    # Testar detecção com imagem simples
    print("\n🎯 Testando detecção...")
    
    try:
        # Criar imagem de teste
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        print("📸 Imagem de teste criada")
        
        # Testar identificação de cartas
        hole_cards, board_cards = analyzer.identify_cards(test_image, debug_mode=False)
        
        print(f"✅ Detecção concluída")
        print(f"   Hole cards: {len(hole_cards)} cartas")
        print(f"   Board cards: {len(board_cards)} cartas")
        
        if hole_cards or board_cards:
            print(f"   Cartas detectadas: {hole_cards + board_cards}")
        else:
            print("   ℹ️ Nenhuma carta detectada (normal para imagem de teste)")
            
    except Exception as e:
        print(f"❌ Erro na detecção: {e}")
        return False
    
    # Testar captura de tela
    print("\n📸 Testando captura de tela...")
    
    try:
        screen_image = analyzer.capture_screen_direct()
        if screen_image is not None:
            print("✅ Captura de tela realizada com sucesso")
            print(f"   Formato da imagem: {screen_image.shape}")
        else:
            print("⚠️ Captura de tela retornou None (ambiente sem display?)")
    except Exception as e:
        print(f"⚠️ Erro na captura de tela: {e}")
    
    # Verificar estatísticas
    print("\n📊 Estatísticas do analisador:")
    
    try:
        stats = analyzer.get_detection_stats() if hasattr(analyzer, 'get_detection_stats') else {}
        for key, value in stats.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"ℹ️ Estatísticas não disponíveis: {e}")
    
    # Testar YOLO específico
    if analyzer.use_yolo and analyzer.yolo_analyzer:
        print("\n🚀 Testando YOLO específico...")
        
        try:
            # Testar com YOLO
            hole_yolo, board_yolo = analyzer.yolo_analyzer.identify_cards(test_image, debug_mode=False)
            print(f"✅ YOLO detecção: Hole={len(hole_yolo)}, Board={len(board_yolo)}")
            
        except Exception as e:
            print(f"⚠️ YOLO detecção falhou: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Teste concluído!")
    
    # Recomendações
    print("\n💡 Recomendações:")
    
    if not analyzer.use_yolo:
        print("   • Para usar YOLO, treine o modelo: python train_yolo11_poker.py")
        print("   • Ou baixe um modelo pré-treinado")
    else:
        print("   ✅ YOLO está funcionando!")
    
    if not (hole_cards or board_cards):
        print("   • O detector não encontrou cartas na imagem de teste")
        print("   • Isso é normal - teste com uma imagem real de cartas")
    
    print("   • Use imagens reais de mesas de poker para testes melhores")
    print("   • Verifique se as cartas estão bem visíveis na tela")
    
    return True

def test_dataset_integrity():
    """Testa integridade do dataset"""
    
    print("\n📊 Testando integridade do dataset...")
    
    dataset_path = Path("cards_templates/IA/dataset")
    
    if not dataset_path.exists():
        print("❌ Dataset não encontrado")
        return False
    
    # Contar imagens e labels
    train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
    train_labels = list((dataset_path / "train" / "labels").glob("*.txt"))
    valid_images = list((dataset_path / "valid" / "images").glob("*.jpg"))
    valid_labels = list((dataset_path / "valid" / "labels").glob("*.txt"))
    
    print(f"✅ Dataset encontrado")
    print(f"   📸 Train images: {len(train_images)}")
    print(f"   🏷️ Train labels: {len(train_labels)}")
    print(f"   📸 Valid images: {len(valid_images)}")
    print(f"   🏷️ Valid labels: {len(valid_labels)}")
    
    # Verificar data.yaml
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        print("✅ data.yaml encontrado")
        
        try:
            import yaml
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"   📊 Classes: {config.get('nc', 'N/A')}")
            print(f"   📝 Nomes: {len(config.get('names', []))} classes")
            
        except Exception as e:
            print(f"⚠️ Erro ao ler data.yaml: {e}")
    else:
        print("❌ data.yaml não encontrado")
    
    return True

if __name__ == "__main__":
    print("🃏 YOLO Poker Integration Test")
    
    # Testar integração
    success = test_yolo_integration()
    
    # Testar dataset
    test_dataset_integrity()
    
    if success:
        print("\n✅ Teste concluído com sucesso!")
        print("\n🚀 Próximos passos:")
        print("1. Treine o modelo: python train_yolo11_poker.py")
        print("2. Teste com imagens reais de cartas")
        print("3. Use na interface: python poker_gui_compact.py")
    else:
        print("\n❌ Teste falhou!")
        print("   Verifique as dependências e tente novamente.")
        
    print("\n" + "=" * 60)