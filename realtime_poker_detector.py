#!/usr/bin/env python3
"""
Poker Real-time YOLO Detector
Integração do YOLOv11 com a interface existente para detecção em tempo real
"""

import cv2
import numpy as np
import threading
import time
import queue
from pathlib import Path
import sys

# Adicionar o diretório atual ao path
sys.path.append(str(Path(__file__).parent))

try:
    from yolo_poker_detector import YOLOPokerDetector, integrate_with_existing_analyzer
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ YOLO Detector não disponível: {e}")
    YOLO_AVAILABLE = False

try:
    from poker_analyzer import PokerAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Poker Analyzer não disponível: {e}")
    ANALYZER_AVAILABLE = False

class RealTimePokerDetector:
    def __init__(self, model_path=None, confidence=0.5, use_fallback=True):
        """
        Detector em tempo real de cartas de poker
        
        Args:
            model_path: Caminho para o modelo YOLOv11 treinado
            confidence: Confiança mínima para detecções
            use_fallback: Usar template matching como fallback
        """
        self.model_path = model_path
        self.confidence = confidence
        self.use_fallback = use_fallback
        
        self.yolo_detector = None
        self.analyzer = None
        self.detection_function = None
        
        # Threading
        self.detection_thread = None
        self.running = False
        self.detection_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Estatísticas
        self.detection_stats = {
            'total_detections': 0,
            'yolo_detections': 0,
            'fallback_detections': 0,
            'avg_confidence': 0.0,
            'detection_time': 0.0
        }
        
        # Buffer para suavização
        self.detection_buffer = []
        self.buffer_size = 3
        
        self.initialize_detectors()
    
    def initialize_detectors(self):
        """Inicializa os detectores YOLO e template matching"""
        
        # Inicializar YOLO detector
        if YOLO_AVAILABLE and self.model_path and Path(self.model_path).exists():
            try:
                self.yolo_detector = YOLOPokerDetector(
                    model_path=self.model_path,
                    confidence=self.confidence
                )
                print(f"✅ YOLO Detector inicializado: {self.model_path}")
            except Exception as e:
                print(f"❌ Erro ao inicializar YOLO Detector: {e}")
                self.yolo_detector = None
        else:
            print("ℹ️ YOLO Detector não disponível ou modelo não encontrado")
            self.yolo_detector = None
        
        # Inicializar Poker Analyzer (template matching)
        if ANALYZER_AVAILABLE and self.use_fallback:
            try:
                self.analyzer = PokerAnalyzer()
                print("✅ Poker Analyzer (template matching) inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar Poker Analyzer: {e}")
                self.analyzer = None
        else:
            self.analyzer = None
        
        # Criar função de detecção integrada
        if self.yolo_detector and self.analyzer:
            self.detection_function = integrate_with_existing_analyzer(
                self.yolo_detector, self.analyzer
            )
        elif self.yolo_detector:
            # Apenas YOLO
            def yolo_only_detection(image, debug_mode=False):
                detections = self.yolo_detector.detect_cards(image)
                analysis = self.yolo_detector.analyze_poker_hand(detections)
                
                # Converter para formato Card (se disponível)
                try:
                    from treys import Card
                    hole_cards = [Card.new(card) for card in analysis['hole_cards']]
                    board_cards = [Card.new(card) for card in analysis['board_cards']]
                    return hole_cards, board_cards
                except ImportError:
                    return analysis['hole_cards'], analysis['board_cards']
            
            self.detection_function = yolo_only_detection
        elif self.analyzer:
            # Apenas template matching
            self.detection_function = lambda image, debug_mode=False: self.analyzer.identify_cards(image, debug_mode)
        else:
            print("❌ Nenhum detector disponível!")
            self.detection_function = None
    
    def start_detection_thread(self):
        """Inicia thread de detecção em tempo real"""
        if not self.detection_function:
            print("❌ Função de detecção não disponível")
            return False
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        print("✅ Thread de detecção iniciada")
        return True
    
    def stop_detection_thread(self):
        """Para thread de detecção"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
            print("✅ Thread de detecção parada")
    
    def _detection_worker(self):
        """Worker thread para processamento contínuo"""
        while self.running:
            try:
                # Obter imagem da fila (timeout de 1 segundo)
                image_data = self.detection_queue.get(timeout=1)
                
                if image_data is None:
                    continue
                
                start_time = time.time()
                
                # Realizar detecção
                hole_cards, board_cards = self.detection_function(image_data, debug_mode=False)
                
                detection_time = time.time() - start_time
                
                # Suavizar detecções
                smoothed_result = self._smooth_detections(hole_cards, board_cards)
                
                # Atualizar estatísticas
                self._update_stats(hole_cards, board_cards, detection_time)
                
                # Colocar resultado na fila
                result = {
                    'hole_cards': smoothed_result['hole_cards'],
                    'board_cards': smoothed_result['board_cards'],
                    'detection_time': detection_time,
                    'timestamp': time.time(),
                    'detector_used': 'YOLO' if self.yolo_detector else 'Template',
                    'confidence': self._get_average_confidence()
                }
                
                # Limpar fila de resultados se estiver cheia
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erro no worker de detecção: {e}")
                time.sleep(0.1)
    
    def detect_cards_realtime(self, image):
        """
        Adiciona imagem para detecção em tempo real
        
        Args:
            image: Imagem OpenCV (BGR)
        
        Returns:
            Dict: Resultado da detecção ou None se não houver
        """
        if not self.running:
            # Iniciar thread se não estiver rodando
            self.start_detection_thread()
        
        # Adicionar imagem à fila de detecção
        if not self.detection_queue.full():
            try:
                self.detection_queue.put(image, block=False)
            except queue.Full:
                pass  # Fila cheia, ignorar
        
        # Obter resultado mais recente
        try:
            result = self.result_queue.get_nowait()
            return result
        except queue.Empty:
            return None
    
    def get_latest_detection(self):
        """Obtém a detecção mais recente sem bloquear"""
        try:
            # Obter todos os resultados disponíveis e retornar o mais recente
            latest_result = None
            while True:
                try:
                    latest_result = self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            return latest_result
        except Exception:
            return None
    
    def _smooth_detections(self, hole_cards, board_cards):
        """Suaviza detecções usando buffer"""
        # Adicionar detecção atual ao buffer
        self.detection_buffer.append({
            'hole_cards': hole_cards,
            'board_cards': board_cards,
            'timestamp': time.time()
        })
        
        # Manter apenas detecções recentes (últimos 3 segundos)
        current_time = time.time()
        self.detection_buffer = [
            det for det in self.detection_buffer 
            if current_time - det['timestamp'] < 3.0
        ]
        
        # Limitar tamanho do buffer
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer = self.detection_buffer[-self.buffer_size:]
        
        # Se não houver detecções suficientes, retornar a mais recente
        if not self.detection_buffer:
            return {'hole_cards': hole_cards, 'board_cards': board_cards}
        
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
        
        return {
            'hole_cards': list(most_common_hole) if most_common_hole else hole_cards,
            'board_cards': list(most_common_board) if most_common_board else board_cards
        }
    
    def _update_stats(self, hole_cards, board_cards, detection_time):
        """Atualiza estatísticas de detecção"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['detection_time'] = detection_time
        
        if self.yolo_detector and self.yolo_detector.model:
            self.detection_stats['yolo_detections'] += 1
        else:
            self.detection_stats['fallback_detections'] += 1
    
    def _get_average_confidence(self):
        """Obtém confiança média das detecções recentes"""
        if self.yolo_detector:
            stats = self.yolo_detector.get_detection_stats()
            return stats.get('avg_confidence', 0.0)
        return 0.0
    
    def get_detection_stats(self):
        """Retorna estatísticas de detecção"""
        stats = self.detection_stats.copy()
        
        # Adicionar informações do detector
        if self.yolo_detector:
            yolo_stats = self.yolo_detector.get_detection_stats()
            stats.update(yolo_stats)
        
        return stats
    
    def draw_detections_on_image(self, image, result, show_stats=True):
        """
        Desenha detecções na imagem
        
        Args:
            image: Imagem OpenCV
            result: Resultado da detecção
            show_stats: Mostrar estatísticas
        
        Returns:
            Image: Imagem com detecções desenhadas
        """
        if not result or not self.yolo_detector:
            return image
        
        # Criar lista de detecções no formato YOLO
        detections = []
        
        # Adicionar hole cards
        for card in result['hole_cards']:
            # Converter para formato de detecção (simulado)
            detections.append({
                'card': str(card),
                'confidence': result.get('confidence', 0.8),
                'bbox': [50, 50, 150, 200]  # Posição simulada
            })
        
        # Adicionar board cards
        for i, card in enumerate(result['board_cards']):
            detections.append({
                'card': str(card),
                'confidence': result.get('confidence', 0.8),
                'bbox': [200 + i*100, 100, 300 + i*100, 250]  # Posição simulada
            })
        
        # Desenhar detecções
        result_image = self.yolo_detector.draw_detections(image, detections)
        
        # Adicionar informações de texto
        if show_stats:
            info_text = f"Hole: {len(result['hole_cards'])} | Board: {len(result['board_cards'])}"
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            time_text = f"Tempo: {result['detection_time']:.3f}s"
            cv2.putText(result_image, time_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return result_image
    
    def cleanup(self):
        """Limpa recursos"""
        self.stop_detection_thread()
        print("🧹 Real-time detector limpo")

# Função auxiliar para integração rápida
def create_realtime_detector(model_path=None, confidence=0.5):
    """
    Cria um detector em tempo real pronto para uso
    
    Args:
        model_path: Caminho opcional para o modelo YOLO
        confidence: Confiança mínima
    
    Returns:
        RealTimePokerDetector: Detector configurado
    """
    return RealTimePokerDetector(
        model_path=model_path,
        confidence=confidence,
        use_fallback=True
    )

# Teste do detector em tempo real
if __name__ == "__main__":
    print("🃏 Real-time Poker Detector - Teste")
    
    # Configurar detector
    model_path = "poker_yolo11/poker_detector_n/weights/best.pt"  # Substitua pelo seu modelo
    
    detector = create_realtime_detector(model_path=model_path)
    
    if not detector.detection_function:
        print("❌ Nenhum detector disponível")
        sys.exit(1)
    
    print("✅ Detector configurado")
    print("💡 Use detector.detect_cards_realtime(image) para detecção em tempo real")
    print("💡 Use detector.get_latest_detection() para obter resultados")
    
    # Exemplo de uso (requer imagem real)
    print("\n📖 Exemplo de uso:")
    print("detector = create_realtime_detector('seu_modelo.pt')")
    print("result = detector.detect_cards_realtime(sua_imagem)")
    print("if result:")
    print("    print(f'Detectado: {result[\"hole_cards\"]} | {result[\"board_cards\"]}')")
    
    # Limpar ao sair
    detector.cleanup()