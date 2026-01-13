#!/usr/bin/env python3
"""
YOLOv11 Poker Real-time Demo
Demonstração do detector YOLOv11 para cartas de poker
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Adicionar diretório atual ao path
sys.path.append(str(Path(__file__).parent))

def create_demo_image():
    """Cria uma imagem de demonstração com cartas de poker"""
    
    # Criar imagem de fundo (simulando mesa de poker)
    img = np.zeros((480, 800, 3), dtype=np.uint8)
    
    # Fundo verde (cor de mesa de poker)
    img[:] = (0, 100, 0)
    
    # Desenhar cartas simuladas
    card_width, card_height = 60, 90
    
    # Hole cards (cartas do jogador)
    hole_positions = [(200, 350), (270, 350)]
    hole_cards = ['As', 'Kd']  # Ás de espadas, Rei de ouros
    
    for i, (x, y) in enumerate(hole_positions):
        # Carta branca
        cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (0, 0, 0), 2)
        
        # Texto da carta
        cv2.putText(img, hole_cards[i], (x + 5, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Board cards (cartas da mesa)
    board_positions = [(150, 200), (230, 200), (310, 200), (390, 200), (470, 200)]
    board_cards = ['Qh', 'Js', 'Tc', '9d', '2h']  # Rainha de copas, Valete de espadas, etc.
    
    for i, (x, y) in enumerate(board_positions):
        # Carta branca
        cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (0, 0, 0), 2)
        
        # Texto da carta
        cv2.putText(img, board_cards[i], (x + 5, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Adicionar informações
    cv2.putText(img, "YOLOv11 Poker Detection Demo", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(img, f"Hole Cards: {', '.join(hole_cards)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(img, f"Board: {', '.join(board_cards)}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img, hole_cards, board_cards

def test_template_matching():
    """Testa o detector de template matching existente"""
    
    print("🃏 Testando Template Matching...")
    
    try:
        from poker_analyzer import PokerAnalyzer
        
        analyzer = PokerAnalyzer(use_yolo=False)  # Apenas template matching
        
        # Criar imagem de teste
        test_image, hole_true, board_true = create_demo_image()
        
        print(f"📸 Imagem de teste criada: {test_image.shape}")
        
        # Testar detecção
        start_time = time.time()
        hole_detected, board_detected = analyzer.identify_cards(test_image, debug_mode=True)
        detection_time = time.time() - start_time
        
        print(f"⏱️ Tempo de detecção: {detection_time:.3f}s")
        print(f"🎯 Hole cards detectadas: {hole_detected}")
        print(f"🃏 Board cards detectadas: {board_detected}")
        
        # Salvar imagem com detecções
        cv2.imwrite("demo_template_detection.png", test_image)
        print(f"💾 Imagem salva: demo_template_detection.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no template matching: {e}")
        return False

def test_yolo_detection():
    """Testa o detector YOLO (se disponível)"""
    
    print("🚀 Testando YOLO Detection...")
    
    try:
        from yolo_poker_detector import YOLOPokerDetector
        
        # Procurar modelo treinado
        model_paths = [
            "poker_yolo11/poker_detector_n/weights/best.pt",
            "poker_yolo11/quick_train_n/weights/best.pt",
            "yolo11n.pt"  # Modelo pré-treinado genérico
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if not model_path:
            print("❌ Nenhum modelo YOLO encontrado")
            print("   Execute: python quick_train_yolo11.py")
            return False
        
        print(f"📁 Usando modelo: {model_path}")
        
        detector = YOLOPokerDetector(model_path)
        
        # Criar imagem de teste
        test_image, hole_true, board_true = create_demo_image()
        
        print(f"📸 Imagem de teste criada: {test_image.shape}")
        
        # Testar detecção YOLO
        start_time = time.time()
        detections = detector.detect_cards(test_image)
        detection_time = time.time() - start_time
        
        print(f"⏱️ Tempo de detecção YOLO: {detection_time:.3f}s")
        print(f"📊 Detecções encontradas: {len(detections)}")
        
        for i, det in enumerate(detections):
            print(f"   {i+1}. {det['card']} (conf: {det['confidence']:.2f})")
        
        # Analisar mão
        analysis = detector.analyze_poker_hand(detections)
        print(f"🎯 Análise da mão: {analysis}")
        
        # Desenhar detecções
        result_image = detector.draw_detections(test_image, detections)
        
        # Salvar imagem com detecções
        cv2.imwrite("demo_yolo_detection.png", result_image)
        print(f"💾 Imagem com detecções salva: demo_yolo_detection.png")
        
        return True
        
    except ImportError:
        print("⚠️ YOLOPokerDetector não disponível")
        return False
    except Exception as e:
        print(f"❌ Erro no YOLO: {e}")
        return False

def test_realtime_detector():
    """Testa o detector em tempo real"""
    
    print("⚡ Testando Real-time Detector...")
    
    try:
        from realtime_poker_detector import create_realtime_detector
        
        # Criar detector em tempo real
        detector = create_realtime_detector()
        
        # Criar imagem de teste
        test_image, hole_true, board_true = create_demo_image()
        
        print(f"📸 Imagem de teste criada: {test_image.shape}")
        
        # Testar detecção em tempo real
        start_time = time.time()
        result = detector.detect_cards_realtime(test_image)
        detection_time = time.time() - start_time
        
        print(f"⏱️ Tempo de detecção real-time: {detection_time:.3f}s")
        
        if result:
            print(f"🎯 Resultado real-time:")
            print(f"   Hole cards: {result['hole_cards']}")
            print(f"   Board cards: {result['board_cards']}")
            print(f"   Tempo: {result['detection_time']:.3f}s")
            print(f"   Detector: {result['detector_used']}")
            
            # Obter estatísticas
            stats = detector.get_detection_stats()
            print(f"📊 Estatísticas: {stats}")
            
            # Desenhar na imagem
            result_image = detector.draw_detections_on_image(test_image, result)
            cv2.imwrite("demo_realtime_detection.png", result_image)
            print(f"💾 Imagem salva: demo_realtime_detection.png")
            
        else:
            print("⚠️ Nenhum resultado obtido (normal se não houver modelo)")
        
        # Limpar
        detector.cleanup()
        
        return True
        
    except ImportError:
        print("⚠️ Realtime detector não disponível")
        return False
    except Exception as e:
        print(f"❌ Erro no real-time detector: {e}")
        return False

def test_integrated_analyzer():
    """Testa o analisador integrado"""
    
    print("🔧 Testando Integrated Analyzer...")
    
    try:
        from yolo_poker_analyzer import YOLOPokerAnalyzer
        
        # Criar analisador integrado
        analyzer = YOLOPokerAnalyzer(use_yolo=False, use_template_fallback=True)
        
        # Criar imagem de teste
        test_image, hole_true, board_true = create_demo_image()
        
        print(f"📸 Imagem de teste criada: {test_image.shape}")
        
        # Testar análise completa
        start_time = time.time()
        hole_detected, board_detected = analyzer.identify_cards(test_image, debug_mode=True)
        detection_time = time.time() - start_time
        
        print(f"⏱️ Tempo de análise: {detection_time:.3f}s")
        print(f"🎯 Hole cards: {hole_detected}")
        print(f"🃏 Board cards: {board_detected}")
        
        # Testar análise de tela completa
        screen_result = analyzer.analyze_screen()
        print(f"📱 Resultado da análise de tela:")
        print(screen_result)
        
        # Estatísticas
        stats = analyzer.get_detection_stats()
        print(f"📊 Estatísticas: {stats}")
        
        return True
        
    except ImportError:
        print("⚠️ YOLOPokerAnalyzer não disponível")
        return False
    except Exception as e:
        print(f"❌ Erro no integrated analyzer: {e}")
        return False

def main():
    """Função principal de demonstração"""
    
    print("🃏 YOLOv11 Poker Detection Demo")
    print("=" * 50)
    
    # Testar diferentes detectores
    tests = [
        ("Template Matching", test_template_matching),
        ("YOLO Detection", test_yolo_detection),
        ("Real-time Detector", test_realtime_detector),
        ("Integrated Analyzer", test_integrated_analyzer),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: SUCESSO")
            else:
                print(f"⚠️ {test_name}: FALHOU (verifique dependências)")
                
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
            results[test_name] = False
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES:")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")
    
    # Recomendações
    print("\n💡 RECOMENDAÇÕES:")
    
    if results.get("Template Matching"):
        print("   ✅ Template matching está funcionando!")
        print("   📁 Verifique a pasta 'unknown_cards' para novos templates")
    
    if not results.get("YOLO Detection"):
        print("   📚 Para usar YOLO:")
        print("      1. Execute: python quick_train_yolo11.py")
        print("      2. Ou baixe um modelo pré-treinado")
    
    print("\n🚀 Próximos passos:")
    print("   1. Teste com imagens reais de mesas de poker")
    print("   2. Use: python poker_gui_compact.py para interface completa")
    print("   3. Ajuste os parâmetros de detecção conforme necessário")
    
    print("\n📁 Arquivos criados:")
    demo_files = ["demo_template_detection.png", "demo_yolo_detection.png", "demo_realtime_detection.png"]
    for file in demo_files:
        if Path(file).exists():
            print(f"   📸 {file}")

if __name__ == "__main__":
    main()