#!/usr/bin/env python3
"""
YOLO Poker Analyzer Integration
Integra o detector YOLOv11 com o poker_analyzer existente
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Adicionar diretório atual ao path
sys.path.append(str(Path(__file__).parent))

try:
    from yolo_poker_detector import YOLOPokerDetector
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ YOLO modules não disponíveis: {e}")
    YOLO_AVAILABLE = False

ANALYZER_AVAILABLE = False

class YOLOPokerAnalyzer:
    """
    Analisador de poker com suporte YOLOv11
    Mantém compatibilidade com poker_analyzer existente mas adiciona YOLO
    """
    
    def __init__(self, yolo_model_path=None, use_yolo=True, use_template_fallback=False):
        """
        Inicializa o analisador YOLO
        
        Args:
            yolo_model_path: Caminho para o modelo YOLO treinado
            use_yolo: Usar detector YOLO
            use_template_fallback: Usar template matching como fallback
        """
        self.yolo_model_path = yolo_model_path
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.use_template_fallback = use_template_fallback and ANALYZER_AVAILABLE
        
        # Detectores
        self.yolo_detector = None
        self.template_analyzer = None
        self.realtime_detector = None
        
        # Configurações
        self.detection_confidence = 0.5
        self.use_detection_smoothing = True
        self.detection_buffer = []
        self.buffer_size = 3
        
        # Estatísticas
        self.stats = {
            'yolo_detections': 0,
            'template_detections': 0,
            'total_analyses': 0,
            'avg_detection_time': 0.0
        }
        
        self.initialize_detectors()
    
    def initialize_detectors(self):
        """Inicializa os detectores disponíveis"""
        
        # Inicializar YOLO
        if self.use_yolo:
            try:
                if self.yolo_model_path and Path(self.yolo_model_path).exists():
                    # Usar modelo específico
                    self.yolo_detector = YOLOPokerDetector(self.yolo_model_path)
                    print(f"✅ YOLO Detector inicializado: {self.yolo_model_path}")
                else:
                    # Tentar usar modelo padrão
                    default_model = Path("poker_yolo11/poker_detector_n/weights/best.pt")
                    if default_model.exists():
                        self.yolo_detector = YOLOPokerDetector(str(default_model))
                        print(f"✅ YOLO Detector inicializado: {default_model}")
                    else:
                        print("⚠️ Modelo YOLO não encontrado, criando detector em tempo real...")
                        from realtime_poker_detector import create_realtime_detector
                        self.realtime_detector = create_realtime_detector()
                        
                        if self.realtime_detector.detection_function:
                            print("✅ Real-time detector criado")
                        else:
                            print("❌ Real-time detector falhou")
                            self.realtime_detector = None
                            self.use_yolo = False
                
            except Exception as e:
                print(f"❌ Erro ao inicializar YOLO: {e}")
                self.use_yolo = False
        
        # Inicializar template matching
        if self.use_template_fallback:
            try:
                self.template_analyzer = PokerAnalyzer()
                print("✅ Template Analyzer inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar template analyzer: {e}")
                self.use_template_fallback = False
        
        # Verificar se temos pelo menos um detector
        if not self.has_any_detector():
            raise RuntimeError("Nenhum detector disponível! Instale ultralytics ou verifique os modelos.")
    
    def has_any_detector(self):
        """Verifica se há algum detector disponível"""
        return (self.yolo_detector is not None or 
                self.realtime_detector is not None or 
                self.template_analyzer is not None)
    
    def identify_cards_yolo(self, image, debug_mode=False):
        """
        Identifica cartas usando YOLO
        
        Args:
            image: Imagem OpenCV
            debug_mode: Modo debug
        
        Returns:
            tuple: (hole_cards, board_cards)
        """
        if not self.use_yolo:
            return [], []
        
        try:
            start_time = datetime.now()
            
            if self.yolo_detector:
                # Usar detector YOLO direto
                detections = self.yolo_detector.detect_cards(image, min_confidence=self.detection_confidence)
                analysis = self.yolo_detector.analyze_poker_hand(detections)
                
                hole_cards_str = analysis['hole_cards']
                board_cards_str = analysis['board_cards']
                
            elif self.realtime_detector:
                # Usar real-time detector
                result = self.realtime_detector.detect_cards_realtime(image)
                
                if result:
                    hole_cards_str = result['hole_cards']
                    board_cards_str = result['board_cards']
                else:
                    hole_cards_str = []
                    board_cards_str = []
            
            else:
                return [], []
            
            # Converter para objetos Card (se disponível)
            try:
                from treys import Card
                hole_cards = [Card.new(card) for card in hole_cards_str]
                board_cards = [Card.new(card) for card in board_cards_str]
            except ImportError:
                # Se treys não estiver disponível, usar strings
                hole_cards = hole_cards_str
                board_cards = board_cards_str
            
            # Atualizar estatísticas
            detection_time = (datetime.now() - start_time).total_seconds()
            self.stats['yolo_detections'] += 1
            self.stats['total_analyses'] += 1
            
            # Suavizar detecções
            if self.use_detection_smoothing:
                hole_cards, board_cards = self._smooth_detections(hole_cards, board_cards)
            
            if debug_mode:
                print(f"🎯 YOLO detectou: Hole={hole_cards}, Board={board_cards}")
                print(f"⏱️ Tempo de detecção: {detection_time:.3f}s")
            
            return hole_cards, board_cards
            
        except Exception as e:
            if debug_mode:
                print(f"❌ Erro na detecção YOLO: {e}")
            return [], []
    
    def identify_cards_template(self, image, debug_mode=False):
        """
        Identifica cartas usando template matching (fallback)
        
        Args:
            image: Imagem OpenCV
            debug_mode: Modo debug
        
        Returns:
            tuple: (hole_cards, board_cards)
        """
        if not self.use_template_fallback or not self.template_analyzer:
            return [], []
        
        try:
            # Usar método original do template analyzer
            return self.template_analyzer.identify_cards(image, debug_mode=debug_mode)
        except Exception as e:
            if debug_mode:
                print(f"❌ Erro na detecção por template: {e}")
            return [], []
    
    def identify_cards(self, image, debug_mode=False):
        """
        Identifica cartas usando YOLO com fallback para template matching
        
        Args:
            image: Imagem OpenCV
            debug_mode: Modo debug
        
        Returns:
            tuple: (hole_cards, board_cards)
        """
        start_time = datetime.now()
        
        # Tentar YOLO primeiro
        if self.use_yolo:
            hole_cards, board_cards = self.identify_cards_yolo(image, debug_mode)
            
            # Se YOLO detectou cartas suficientes, usar essas
            if len(hole_cards) >= 2:
                if debug_mode:
                    print(f"✅ YOLO detectou {len(hole_cards)} hole cards e {len(board_cards)} board cards")
                return hole_cards, board_cards
            
            # Se YOLO falhou e temos fallback, tentar template matching
            elif self.use_template_fallback:
                if debug_mode:
                    print("🔄 YOLO não detectou suficiente, tentando template matching...")
                return self.identify_cards_template(image, debug_mode)
        
        # Apenas template matching
        elif self.use_template_fallback:
            return self.identify_cards_template(image, debug_mode)
        
        # Nenhum detector disponível
        return [], []
    
    def _smooth_detections(self, hole_cards, board_cards):
        """Suaviza detecções usando buffer"""
        # Adicionar detecção atual ao buffer
        self.detection_buffer.append({
            'hole_cards': hole_cards,
            'board_cards': board_cards,
            'timestamp': datetime.now()
        })
        
        # Limitar tamanho do buffer
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer = self.detection_buffer[-self.buffer_size:]
        
        # Se não houver detecções suficientes, retornar a mais recente
        if not self.detection_buffer:
            return hole_cards, board_cards
        
        # Usar a detecção mais comum (moda)
        hole_cards_counts = {}
        board_cards_counts = {}
        
        for detection in self.detection_buffer:
            # Contar hole cards
            hole_key = tuple(sorted(detection['hole_cards']))
            hole_cards_counts[hole_key] = hole_cards_counts.get(hole_key, 0) + 1
            
            # Contar board cards
            board_key = tuple(sorted(detection['board_cards']))
            board_cards_counts[board_key] = board_cards_counts.get(board_key, 0) + 1
        
        # Encontrar as detecções mais comuns
        most_common_hole = max(hole_cards_counts.items(), key=lambda x: x[1])[0] if hole_cards_counts else hole_cards
        most_common_board = max(board_cards_counts.items(), key=lambda x: x[1])[0] if board_cards_counts else board_cards
        
        return list(most_common_hole), list(most_common_board)
    
    def capture_screen_direct(self):
        """
        Captura tela diretamente (mantém compatibilidade)
        """
        if self.template_analyzer and hasattr(self.template_analyzer, 'capture_screen_direct'):
            return self.template_analyzer.capture_screen_direct()
        else:
            # Fallback básico
            try:
                from PIL import ImageGrab
                import numpy as np
                
                screenshot = ImageGrab.grab()
                img = np.array(screenshot)
                
                # Converter RGB para BGR (formato OpenCV)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                return img
                
            except ImportError:
                print("❌ PIL não disponível para captura de tela")
                return None
            except Exception as e:
                print(f"❌ Erro na captura de tela: {e}")
                return None
    
    def analyze_screen(self):
        """
        Analisa a tela completa (mantém compatibilidade)
        """
        try:
            # Capturar tela
            img = self.capture_screen_direct()
            if img is None:
                return "Erro: Não foi possível capturar a tela"
            
            # Identificar cartas
            hole, board = self.identify_cards(img, debug_mode=True)
            
            if len(hole) < 2:
                return "⚠️ Cartas insuficientes na mão (preciso de pelo menos 2)"
            
            # Calcular equidade (se disponível)
            equity = 0.0
            if self.template_analyzer and hasattr(self.template_analyzer, 'calculate_equity'):
                equity = self.template_analyzer.calculate_equity(hole, board)
            
            # Formatar resultado
            result = f"🎯 ANÁLISE COMPLETA:\n"
            result += f"🂠 Suas cartas: {[str(c) for c in hole]}\n"
            result += f"🃏 Mesa: {[str(c) for c in board]}\n"
            result += f"📊 Equidade: {equity*100:.1f}%\n"
            result += f"💡 Detector: {'YOLO' if self.use_yolo else 'Template'}"
            
            return result
            
        except Exception as e:
            return f"❌ Erro na análise: {str(e)}"
    
    def get_detection_stats(self):
        """Retorna estatísticas de detecção"""
        stats = self.stats.copy()
        
        # Adicionar informações sobre detectores
        stats['yolo_available'] = self.use_yolo
        stats['template_available'] = self.use_template_fallback
        stats['detector_used'] = 'YOLO' if self.use_yolo else 'Template'
        
        return stats
    
    def set_detection_confidence(self, confidence):
        """Define a confiança mínima para detecções"""
        self.detection_confidence = max(0.0, min(1.0, confidence))
    
    def enable_detection_smoothing(self, enable=True):
        """Habilita/desabilita suavização de detecções"""
        self.use_detection_smoothing = enable

# Função de conveniência para criar analisador YOLO
def create_yolo_analyzer(yolo_model_path=None, use_template_fallback=False):
    """
    Cria um analisador YOLO configurado
    
    Args:
        yolo_model_path: Caminho para o modelo YOLO
        use_template_fallback: Usar template matching como fallback
    
    Returns:
        YOLOPokerAnalyzer: Analisador configurado
    """
    return YOLOPokerAnalyzer(
        yolo_model_path=yolo_model_path,
        use_yolo=True,
        use_template_fallback=use_template_fallback
    )

# Teste do analisador YOLO
if __name__ == "__main__":
    print("🃏 YOLO Poker Analyzer - Teste")
    print("=" * 50)
    
    # Configurar analisador
    yolo_model = "poker_yolo11/poker_detector_n/weights/best.pt"  # Modelo padrão
    
    try:
        analyzer = create_yolo_analyzer(yolo_model_path=yolo_model)
        
        print("✅ Analisador YOLO criado com sucesso!")
        print(f"📊 Stats: {analyzer.get_detection_stats()}")
        
        # Teste rápido
        print("\n🧪 Testando detecção...")
        
        # Criar imagem de teste (ou usar uma real)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        hole, board = analyzer.identify_cards(test_image, debug_mode=True)
        
        print(f"Resultado: Hole={hole}, Board={board}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("💡 Verifique se o modelo YOLO está treinado ou use apenas template matching")
        
        # Tentar com template apenas
        if ANALYZER_AVAILABLE:
            print("\n🔄 Tentando com template matching apenas...")
            analyzer = YOLOPokerAnalyzer(use_yolo=False, use_template_fallback=True)
            print("✅ Analisador template criado com sucesso!")
        else:
            print("❌ Nenhum analisador disponível")
