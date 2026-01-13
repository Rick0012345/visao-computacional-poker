#!/usr/bin/env python3
"""
YOLOv11 Poker Card Detector
Detecção em tempo real de cartas de poker usando YOLOv11
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
import threading
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Adicionar o diretório do YOLOv11 ao path
try:
    # Tentar importar ultralytics (para YOLOv8/v11)
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics não disponível. Instale com: pip install ultralytics")

class YOLOPokerDetector:
    def __init__(self, model_path=None, confidence=0.5, iou_threshold=0.45):
        """
        Inicializa o detector YOLOv11 para cartas de poker
        
        Args:
            model_path: Caminho para o modelo YOLOv11 treinado
            confidence: Threshold de confiança para detecções
            iou_threshold: Threshold para NMS (Non-Maximum Suppression)
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        self.classes = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Mapeamento de classes para cartas
        self.card_mapping = {
            0: '10c', 1: '10d', 2: '10h', 3: '10s',
            4: '2c', 5: '2d', 6: '2h', 7: '2s',
            8: '3c', 9: '3d', 10: '3h', 11: '3s',
            12: '4c', 13: '4d', 14: '4h', 15: '4s',
            16: '5c', 17: '5d', 18: '5h', 19: '5s',
            20: '6c', 21: '6d', 22: '6h', 23: '6s',
            24: '7c', 25: '7d', 26: '7h', 27: '7s',
            28: '8c', 29: '8d', 30: '8h', 31: '8s',
            32: '9c', 33: '9d', 34: '9h', 35: '9s',
            36: 'Ac', 37: 'Ad', 38: 'Ah', 39: 'As',
            40: 'Jc', 41: 'Jd', 42: 'Jh', 43: 'Js',
            44: 'Kc', 45: 'Kd', 46: 'Kh', 47: 'Ks',
            48: 'Qc', 49: 'Qd', 50: 'Qh', 51: 'Qs'
        }
        
        # Buffer para detecções suavizadas
        self.detection_buffer = deque(maxlen=5)
        self.last_detection_time = 0
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("ℹ️ Nenhum modelo fornecido. Use train_model() para treinar ou forneça um modelo pré-treinado.")
    
    def load_model(self, model_path):
        """Carrega o modelo YOLOv11"""
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("Ultralytics não está instalado")
            
            self.model = YOLO(model_path)
            self.classes = self.model.names
            print(f"✅ Modelo YOLO carregado: {model_path}")
            print(f"📱 Dispositivo: {self.device}")
            print(f"📊 Classes disponíveis: {len(self.classes)}")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
    
    def train_model(self, data_yaml, epochs=100, imgsz=640, batch_size=16):
        """
        Treina um novo modelo YOLOv11 com o dataset fornecido
        
        Args:
            data_yaml: Caminho para o arquivo data.yaml com configurações do dataset
            epochs: Número de épocas para treinamento
            imgsz: Tamanho das imagens
            batch_size: Tamanho do batch
        """
        if not YOLO_AVAILABLE:
            print("❌ Ultralytics não está disponível. Instale com: pip install ultralytics")
            return False
        
        try:
            print("🚀 Iniciando treinamento do modelo YOLOv11...")
            
            # Criar modelo a partir do zero ou usar um pré-treinado
            model = YOLO('yolo11n.pt')  # Usar modelo pré-treinado nano
            
            # Treinar o modelo
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=self.device,
                patience=20,  # Early stopping
                save=True,
                save_period=10,
                project='poker_yolo11',
                name='poker_detector',
                exist_ok=True
            )
            
            print("✅ Treinamento concluído!")
            self.model = model
            return True
            
        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            return False
    
    def detect_cards(self, image, min_confidence=None):
        """
        Detecta cartas em uma imagem
        
        Args:
            image: Imagem OpenCV (BGR) ou caminho para imagem
            min_confidence: Confiança mínima (usa o padrão se None)
        
        Returns:
            List: Lista de detecções [x1, y1, x2, y2, confiança, classe]
        """
        if self.model is None:
            print("❌ Modelo não carregado")
            return []
        
        if min_confidence is None:
            min_confidence = self.confidence
        
        try:
            # Converter imagem se necessário
            if isinstance(image, str):
                image = cv2.imread(image)
            
            # Fazer predição
            results = self.model(image, conf=min_confidence, iou=self.iou_threshold)
            
            # Extrair detecções
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': cls,
                            'card': self.card_mapping.get(cls, f'unknown_{cls}')
                        })
            
            # Adicionar ao buffer para suavização
            self.detection_buffer.append(detections)
            self.last_detection_time = time.time()
            
            return detections
            
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
            return []
    
    def get_stable_detections(self, min_confidence=None):
        """
        Obtém detecções estabilizadas usando média do buffer
        
        Returns:
            List: Detecções estabilizadas
        """
        if not self.detection_buffer:
            return []
        
        # Agrupar detecções por proximidade
        all_detections = []
        for detections in self.detection_buffer:
            all_detections.extend(detections)
        
        # Filtrar por confiança mínima
        if min_confidence:
            all_detections = [d for d in all_detections if d['confidence'] >= min_confidence]
        
        return all_detections
    
    def draw_detections(self, image, detections, show_confidence=True, show_label=True):
        """
        Desenha as detecções na imagem
        
        Args:
            image: Imagem OpenCV
            detections: Lista de detecções
            show_confidence: Mostrar confiança
            show_label: Mostrar label da carta
        
        Returns:
            Image: Imagem com detecções desenhadas
        """
        img_copy = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            card = detection['card']
            
            # Desenhar retângulo
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Preparar texto
            label_parts = []
            if show_label:
                label_parts.append(card)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Desenhar fundo do texto
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (bbox[0], bbox[1] - text_height - 10), 
                         (bbox[0] + text_width, bbox[1]), (0, 255, 0), -1)
            
            # Desenhar texto
            cv2.putText(img_copy, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy
    
    def analyze_poker_hand(self, detections):
        """
        Analisa as cartas detectadas para identificar mãos de poker
        
        Args:
            detections: Lista de detecções
        
        Returns:
            Dict: Análise da mão {'hole_cards': [], 'board_cards': [], 'hand_strength': ''}
        """
        if not detections:
            return {'hole_cards': [], 'board_cards': [], 'hand_strength': 'Nenhuma carta detectada'}
        
        # Separar cartas por posição (simplificado - pode ser melhorado)
        # Assumindo que as 2 primeiras são hole cards e o resto é board
        cards = [d['card'] for d in detections]
        
        hole_cards = cards[:2]  # 2 primeiras cartas
        board_cards = cards[2:]  # Restante é board
        
        # Análise básica da força da mão
        hand_strength = self.evaluate_hand_strength(hole_cards, board_cards)
        
        return {
            'hole_cards': hole_cards,
            'board_cards': board_cards,
            'hand_strength': hand_strength,
            'total_cards': len(cards)
        }
    
    def evaluate_hand_strength(self, hole_cards, board_cards):
        """
        Avalia a força da mão de poker (simplificado)
        
        Args:
            hole_cards: Cartas do jogador
            board_cards: Cartas da mesa
        
        Returns:
            str: Descrição da força da mão
        """
        all_cards = hole_cards + board_cards
        
        if len(all_cards) < 2:
            return "Muito poucas cartas"
        
        # Análise muito básica - pode ser melhorada com lógica real de poker
        card_values = [card[:-1] for card in all_cards]
        card_suits = [card[-1] for card in all_cards]
        
        # Contar pares
        value_counts = {}
        for value in card_values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        pairs = sum(1 for count in value_counts.values() if count >= 2)
        
        # Contar cartas do mesmo naipe
        suit_counts = {}
        for suit in card_suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_suit = max(suit_counts.values()) if suit_counts else 0
        
        # Análise simplificada
        if len(all_cards) >= 5:
            if pairs >= 2:
                return "Dois pares ou melhor"
            elif pairs == 1:
                return "Um par"
            elif max_suit >= 5:
                return "Flush possível"
            else:
                return "Carta alta"
        elif len(hole_cards) == 2:
            if hole_cards[0][:-1] == hole_cards[1][:-1]:
                return "Par nas hole cards!"
            elif hole_cards[0][-1] == hole_cards[1][-1]:
                return "Mesmo naipe nas hole cards"
            else:
                return "Hole cards normais"
        else:
            return "Mão incompleta"
    
    def get_detection_stats(self):
        """
        Retorna estatísticas das detecções recentes
        
        Returns:
            Dict: Estatísticas de detecção
        """
        if not self.detection_buffer:
            return {'avg_confidence': 0, 'total_detections': 0, 'unique_cards': 0}
        
        all_detections = []
        for detections in self.detection_buffer:
            all_detections.extend(detections)
        
        if not all_detections:
            return {'avg_confidence': 0, 'total_detections': 0, 'unique_cards': 0}
        
        avg_confidence = sum(d['confidence'] for d in all_detections) / len(all_detections)
        total_detections = len(all_detections)
        unique_cards = len(set(d['card'] for d in all_detections))
        
        return {
            'avg_confidence': avg_confidence,
            'total_detections': total_detections,
            'unique_cards': unique_cards
        }

# Funções auxiliares para integração com o poker_analyzer existente
def create_yolo_detector(model_path=None):
    """
    Cria uma instância do detector YOLO
    
    Args:
        model_path: Caminho opcional para o modelo
    
    Returns:
        YOLOPokerDetector: Instância do detector
    """
    return YOLOPokerDetector(model_path)

def integrate_with_existing_analyzer(yolo_detector, poker_analyzer):
    """
    Integra o detector YOLO com o PokerAnalyzer existente
    
    Args:
        yolo_detector: Instância YOLOPokerDetector
        poker_analyzer: Instância PokerAnalyzer existente
    
    Returns:
        Função de detecção integrada
    """
    def integrated_detection(image, debug_mode=False):
        """
        Função de detecção integrada que usa YOLO e fallback para template matching
        """
        # Tentar detecção YOLO primeiro
        yolo_detections = yolo_detector.detect_cards(image)
        
        if yolo_detections and len(yolo_detections) >= 2:
            # YOLO detectou cartas suficientes
            analysis = yolo_detector.analyze_poker_hand(yolo_detections)
            
            # Converter para formato do poker_analyzer existente
            hole_cards = analysis['hole_cards']
            board_cards = analysis['board_cards']
            
            # Converter strings para objetos Card do treys
            try:
                from treys import Card
                hole_cards_obj = [Card.new(card) for card in hole_cards]
                board_cards_obj = [Card.new(card) for card in board_cards]
                
                if debug_mode:
                    print(f"🎯 YOLO detectou: Hole={hole_cards}, Board={board_cards}")
                
                return hole_cards_obj, board_cards_obj
            except ImportError:
                # Se treys não estiver disponível, retornar strings
                return hole_cards, board_cards
        
        else:
            # Fallback para template matching do poker_analyzer existente
            if debug_mode:
                print("🔄 YOLO não detectou suficiente, usando template matching...")
            
            # Usar método existente como fallback
            if hasattr(poker_analyzer, 'identify_cards'):
                return poker_analyzer.identify_cards(image, debug_mode)
            else:
                return [], []
    
    return integrated_detection

# Teste do detector
if __name__ == "__main__":
    print("🃏 YOLO Poker Card Detector - Teste")
    
    # Criar detector
    detector = create_yolo_detector()
    
    if detector.model is None:
        print("❌ Modelo não disponível. Treine o modelo primeiro.")
        sys.exit(1)
    
    # Teste com imagem de exemplo
    test_image_path = "test_poker_table.png"  # Substitua por uma imagem real
    
    if os.path.exists(test_image_path):
        print(f"📸 Testando com imagem: {test_image_path}")
        
        # Carregar imagem
        image = cv2.imread(test_image_path)
        
        # Detectar cartas
        detections = detector.detect_cards(image)
        
        if detections:
            print(f"✅ Detectou {len(detections)} cartas:")
            for det in detections:
                print(f"  - {det['card']} (conf: {det['confidence']:.2f})")
            
            # Analisar mão
            analysis = detector.analyze_poker_hand(detections)
            print(f"📊 Análise: {analysis}")
            
            # Desenhar detecções
            result_image = detector.draw_detections(image, detections)
            cv2.imwrite("detection_result.png", result_image)
            print("🖼️ Resultado salvo em: detection_result.png")
            
        else:
            print("❌ Nenhuma carta detectada")
    
    else:
        print(f"❌ Imagem de teste não encontrada: {test_image_path}")
        print("💡 Crie uma imagem de teste ou use o detector em tempo real")