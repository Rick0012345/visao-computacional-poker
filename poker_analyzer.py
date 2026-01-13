import mss
import mss.tools
import cv2
import numpy as np
from PIL import Image, ImageGrab
from treys import Card, Deck, Evaluator
import time
import os
import glob
import pyautogui
import winsound
import ctypes
import sys

# YOLO Integration (opcional)
try:
    from yolo_poker_analyzer import YOLOPokerAnalyzer, create_yolo_analyzer
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class PokerAnalyzer:
    def __init__(self, use_yolo=True, yolo_model_path=None):
        """
        Inicializa o PokerAnalyzer com suporte opcional YOLO
        
        Args:
            use_yolo: Habilitar detector YOLO
            yolo_model_path: Caminho opcional para modelo YOLO
        """
        # Garante que as pastas necessárias existem
        for directory in ["cards_templates", "unknown_cards"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"[Info] Pasta criada: {directory}")

        self.templates = {}
        self.load_templates()
        
        # Inicializar YOLO detector (opcional)
        self.yolo_analyzer = None
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        
        if self.use_yolo:
            try:
                self.yolo_analyzer = create_yolo_analyzer(yolo_model_path)
                print("[Info] YOLO Poker Analyzer inicializado")
            except Exception as e:
                print(f"[Aviso] Falha ao inicializar YOLO: {e}")
                self.use_yolo = False
        
        # Configurações de captura de tela
        try:
            self.sct = mss.mss()
            self.has_display = True
        except Exception:
            self.has_display = False
            print("[Aviso] Ambiente sem display detectado. Usando modo de simulação de captura.")

    def load_templates(self):
        """
        Carrega os templates de cartas da pasta 'cards_templates'.
        Agora suporta estrutura de pastas organizadas:
        cards_templates/
        ├── As/
        │   ├── As_1.png
        │   ├── As_2.png
        │   └── As_3.png
        ├── Kh/
        │   ├── Kh_1.png
        │   └── Kh_2.png
        └── etc...
        
        O nome da pasta é o código da carta (ex: 'As', 'Kd.png', 'Th.png').
        """
        self.templates = {}  # Limpa templates anteriores
        
        # Verifica se a pasta cards_templates existe
        if not os.path.exists("cards_templates"):
            print("[Info] Pasta 'cards_templates' não encontrada. Criando estrutura...")
            os.makedirs("cards_templates", exist_ok=True)
            return
        
        # Percorre todas as subpastas em cards_templates
        for carta_dir in os.listdir("cards_templates"):
            carta_path = os.path.join("cards_templates", carta_dir)
            
            # Verifica se é uma pasta (e não um arquivo .png solto)
            if os.path.isdir(carta_path):
                carta_nome = carta_dir  # Nome da pasta = nome da carta
                print(f"[Debug] Carregando templates para {carta_nome}...")
                
                # Carrega todas as imagens da pasta
                imagens_carta = []
                for arquivo in os.listdir(carta_path):
                    if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(carta_path, arquivo)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            imagens_carta.append(img)
                            print(f"  ✓ Carregado: {arquivo}")
                        else:
                            print(f"  ✗ Falha ao carregar: {arquivo}")
                
                if imagens_carta:
                    # Armazena a lista de templates para esta carta
                    self.templates[carta_nome] = imagens_carta
                    print(f"[Info] {len(imagens_carta)} templates carregados para {carta_nome}")
                else:
                    print(f"[Aviso] Nenhuma imagem válida encontrada em {carta_path}")
            
            # Também suporta arquivos .png soltos para compatibilidade retro
            elif carta_dir.endswith('.png'):
                carta_nome = carta_dir[:-4]  # Remove .png
                img_path = os.path.join("cards_templates", carta_dir)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[carta_nome] = [img]  # Lista com um único template
                    print(f"[Info] Template único carregado: {carta_nome}")
        
        print(f"\n[Info] Total de cartas com templates: {len(self.templates)}")
        if self.templates:
            print("Cartas carregadas:", list(self.templates.keys()))

    def capture_screen(self, region=None):
        """
        Captura a tela simulando comportamento humano: usa Alt+PrintScreen para 
        capturar a janela ativa, exatamente como você faria manualmente.
        """
        if not self.has_display:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
            
        try:
            # Método 1: Simula Alt+PrintScreen (captura janela ativa)
            print("[Debug] Simulando Alt+PrintScreen...")
            
            # Limpa o clipboard
            try:
                ImageGrab.grabclipboard()
            except:
                pass
                
            # Pressiona Alt+PrintScreen
            pyautogui.hotkey('alt', 'printscreen')
            time.sleep(0.5)  # Espera o Windows processar
            
            # Pega imagem do clipboard
            img = ImageGrab.grabclipboard()
            
            if img:
                print("[Debug] Captura via Alt+PrintScreen funcionou!")
                # Som de câmera para confirmar captura
                winsound.Beep(1000, 200)  # Beep agudo de 200ms
                return np.array(img)
            else:
                print("[Debug] Alt+PrintScreen não conseguiu imagem. Tentando método 2...")
                
        except Exception as e:
            print(f"[Debug] Erro no Alt+PrintScreen: {e}")
        
        # Método 2: PrintScreen total (se Alt+Print falhou)
        try:
            print("[Debug] Tentando PrintScreen total...")
            pyautogui.hotkey('printscreen')
            time.sleep(0.5)
            
            img = ImageGrab.grabclipboard()
            if img:
                print("[Debug] PrintScreen total funcionou!")
                winsound.Beep(1000, 200)  # Beep agudo de 200ms
                return np.array(img)
            else:
                print("[Debug] PrintScreen total também falhou...")
                
        except Exception as e:
            print(f"[Debug] Erro no PrintScreen: {e}")
        
        # Método 3: Captura direta (último recurso)
        print("[Debug] Tentando captura direta com pyautogui...")
        try:
            screenshot = pyautogui.screenshot()
            print("[Debug] Captura direta funcionou!")
            winsound.Beep(1000, 200)  # Beep agudo de 200ms
            return np.array(screenshot)
        except Exception as e:
            print(f"[Erro] Todos os métodos falharam: {e}")
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def capture_screen_region(self, x, y, width, height):
        """
        Captura apenas uma região específica da tela (útil para pegar só as cartas).
        """
        if not self.has_display:
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        try:
            # Usa MSS para captura rápida de região
            monitor = {"left": x, "top": y, "width": width, "height": height}
            screenshot = self.sct.grab(monitor)
            img = np.array(screenshot)
            # Converte de BGRA para BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            print(f"[Debug] Erro na captura de região: {e}")
            # Fallback para Pillow
            try:
                img = ImageGrab.grab(bbox=(x, y, x+width, y+height))
                return np.array(img)
            except Exception as e2:
                print(f"[Debug] Fallback Pillow também falhou: {e2}")
                return np.zeros((height, width, 3), dtype=np.uint8)

    def run_capture_diagnostics(self):
        """
        Executa um teste de diagnóstico capturando a tela com 3 métodos diferentes
        e salvando os resultados para o usuário verificar qual funciona.
        """
        print("\n[Diagnóstico] Iniciando teste de captura de tela...")
        print("[Diagnóstico] Vou salvar 3 imagens: diag_pyautogui.png, diag_pillow.png, diag_mss.png")
        print("[Diagnóstico] Verifique qual delas mostra a janela do GG Poker corretamente.")
        
        # 1. Teste PyAutoGUI
        try:
            print("[Diagnóstico] Tentando PyAutoGUI...")
            shot1 = pyautogui.screenshot()
            shot1.save("diag_pyautogui.png")
            print(" -> Sucesso (Salvo diag_pyautogui.png)")
        except Exception as e:
            print(f" -> Falha: {e}")

        # 2. Teste Pillow (ImageGrab)
        try:
            print("[Diagnóstico] Tentando Pillow (ImageGrab)...")
            shot2 = ImageGrab.grab()
            shot2.save("diag_pillow.png")
            print(" -> Sucesso (Salvo diag_pillow.png)")
        except Exception as e:
            print(f" -> Falha: {e}")

        # 3. Teste MSS
        try:
            print("[Diagnóstico] Tentando MSS...")
            monitor = self.sct.monitors[1]
            shot3 = self.sct.grab(monitor)
            mss.tools.to_png(shot3.rgb, shot3.size, output="diag_mss.png")
            print(" -> Sucesso (Salvo diag_mss.png)")
        except Exception as e:
            print(f" -> Falha: {e}")
            
        print("[Diagnóstico] Teste concluído. Verifique as imagens na pasta.\n")
        # Beep para avisar que o diagnóstico terminou
        winsound.Beep(1000, 500)

    def identify_cards(self, screen_image, debug_mode=True):
        """
        Identifica cartas na imagem da tela usando YOLO (se disponível) ou Template Matching.
        Suporta múltiplos templates por carta para melhor reconhecimento.
        """
        # Tentar YOLO primeiro (se disponível e habilitado)
        if self.use_yolo and self.yolo_analyzer:
            try:
                hole_cards, board_cards = self.yolo_analyzer.identify_cards(screen_image, debug_mode)
                if len(hole_cards) >= 2:  # YOLO detectou cartas suficientes
                    if debug_mode:
                        print(f"[Info] YOLO detectou: Hole={len(hole_cards)}, Board={len(board_cards)}")
                    return hole_cards, board_cards
                elif debug_mode:
                    print("[Info] YOLO não detectou cartas suficientes, tentando template matching...")
            except Exception as e:
                if debug_mode:
                    print(f"[Aviso] YOLO falhou: {e}, usando template matching...")
        
        # Fallback para template matching
        if not self.templates:
            print("[Aviso] Nenhum template encontrado. Entrando em modo de COLETA DE TEMPLATES.")
            self.extract_potential_cards(screen_image)
            return [], []
        
        # Salvar imagem original para debug se solicitado
        if debug_mode:
            try:
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_filename = f"debug_identify_cards_{timestamp}.png"
                cv2.imwrite(debug_filename, screen_image)
                print(f"[Debug] Imagem original salva: {debug_filename} em {os.getcwd()}")
            except Exception as e:
                print(f"[Erro] Falha ao salvar imagem debug: {e}")

        gray_screen = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        found_cards = []

        # Para cada carta, testa todos os seus templates
        for carta_nome, template_list in self.templates.items():
            melhor_match = 0
            template_usado = None
            template_tamanho = None
            
            # Testa cada template da carta
            for template in template_list:
                h, w = template.shape
                template_tamanho = f"{w}x{h}"
                
                # Ajusta threshold baseado no tamanho do template
                if w < 30 or h < 50:  # Templates pequenos (como 22x53)
                    threshold_minimo = 0.90  # 90% para templates pequenos
                elif w < 50 or h < 70:  # Templates médios
                    threshold_minimo = 0.85  # 85% para templates médios
                else:  # Templates grandes
                    threshold_minimo = 0.80  # 80% para templates grandes
                
                res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > melhor_match:
                    melhor_match = max_val
                    template_usado = template
            
            # Validação adicional para templates pequenos
            if w < 30 or h < 50:  # Templates pequenos precisam de validação extra
                # Conta quantos templates da mesma carta atingem threshold alto
                templates_validos = 0
                for template_check in template_list:
                    res_check = cv2.matchTemplate(gray_screen, template_check, cv2.TM_CCOEFF_NORMED)
                    _, max_val_check, _, _ = cv2.minMaxLoc(res_check)
                    if max_val_check >= 0.85:  # Threshold mais alto para validação cruzada
                        templates_validos += 1
                
                # Para templates pequenos, requer pelo menos 2 templates confirmando
                if templates_validos >= 2 and melhor_match >= threshold_minimo:
                    found_cards.append(carta_nome)
                    print(f"[Debug] Carta {carta_nome} reconhecida com {melhor_match*100:.1f}% (validação cruzada: {templates_validos} templates)")
                elif melhor_match >= 0.70:
                    print(f"[Debug] Carta {carta_nome} próxima: {melhor_match*100:.1f}% (faltou validação cruzada)")
            else:
                # Templates grandes usam threshold normal
                if melhor_match >= threshold_minimo:
                    found_cards.append(carta_nome)
                    print(f"[Debug] Carta {carta_nome} reconhecida com {melhor_match*100:.1f}% de confiança (template: {template_tamanho}, threshold: {threshold_minimo*100:.0f}%)")
                elif melhor_match >= 0.70:  # Debug: mostra quando está próximo mas não passou
                    print(f"[Debug] Carta {carta_nome} próxima: {melhor_match*100:.1f}% (template: {template_tamanho}, precisava: {threshold_minimo*100:.0f}%)")
        
        # Remove duplicatas (se houver)
        unique_cards = list(set(found_cards))
        
        # Converte para objetos Card do treys
        valid_cards = []
        for c in unique_cards:
            try:
                valid_cards.append(Card.new(c))
            except Exception:
                pass # Ignora nomes inválidos

        # Separa hole cards e board cards
        if len(valid_cards) >= 2:
            hole_cards = valid_cards[:2]
            board_cards = valid_cards[2:]
        else:
            hole_cards = valid_cards
            board_cards = []
            
        return hole_cards, board_cards


    def extract_potential_cards(self, screen_image):
        """
        Tenta encontrar contornos retangulares que pareçam cartas e os salva para o usuário nomear.
        """
        gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold para binarizar (pode precisar ajustar dependendo do fundo da mesa)
        # Tenta OTSU para encontrar o threshold ótimo automaticamente
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Salva imagem binarizada para debug
        try:
            timestamp = int(time.time())
            debug_filename = f"debug_threshold_{timestamp}.png"
            cv2.imwrite(debug_filename, thresh)
            print(f"[Debug] Imagem {debug_filename} salva em: {os.getcwd()}")
        except Exception as e:
            print(f"[Erro] Falha ao salvar {debug_filename}: {e}")
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Imagem para desenhar os contornos (debug visual)
        debug_img = screen_image.copy()
        
        saved_count = 0
        print(f"[Debug] Encontrei {len(contours)} contornos iniciais. Filtrando...")
        
        for i, cnt in enumerate(contours):
            # Desenha todos os contornos em azul (fino) para saber que foi detectado
            cv2.drawContours(debug_img, [cnt], -1, (255, 0, 0), 1)
            
            # Filtra por área (Relaxado para GG Poker: cartas podem ser menores ou maiores)
            area = cv2.contourArea(cnt)
            if 500 < area < 100000: 
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                
                # Cartas GG Poker são inclinadas (ratio muda) e podem estar sobrepostas
                # Relaxando ratio: de 0.5 (carta em pé) até 1.5 (carta deitada/inclinada ou duas juntas)
                if 0.5 < aspect_ratio < 2.0:
                    # Desenha retângulo verde nos aceitos
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    roi = screen_image[y:y+h, x:x+w]
                    timestamp = int(time.time())
                    filename = f"unknown_cards/card_{timestamp}_{i}.png"
                    cv2.imwrite(filename, roi)
                    saved_count += 1
        
        # Salva o mapa de contornos
        try:
            timestamp = int(time.time())
            debug_filename = f"debug_contours_{timestamp}.png"
            cv2.imwrite(debug_filename, debug_img)
            print(f"[Debug] Salvei '{debug_filename}' em: {os.getcwd()}")
            print("[Debug] Imagem mostra o que foi detectado (Verde = Salvo, Azul = Ignorado).")
        except Exception as e:
            print(f"[Erro] Falha ao salvar {debug_filename}: {e}")
        
        # Criar diretório unknown_cards se não existir
        os.makedirs("unknown_cards", exist_ok=True)
        
        if saved_count > 0:
            print(f"[Info] {saved_count} potenciais cartas salvas na pasta 'unknown_cards'.")
            print("POR FAVOR: Vá até a pasta 'unknown_cards', identifique as cartas e mova/renomeie para 'cards_templates' (ex: 'As.png').")
        else:
            print("[Aviso] Não consegui detectar contornos de cartas claros. Tente ajustar a iluminação ou o threshold.")
            # Salva a tela inteira para debug
            try:
                timestamp = int(time.time())
                debug_filename = f"debug_screen_{timestamp}.png"
                cv2.imwrite(debug_filename, screen_image)
                print(f"[Debug] Salvei '{debug_filename}' em: {os.getcwd()}")
            except Exception as e:
                print(f"[Erro] Falha ao salvar {debug_filename}: {e}")


    def calculate_equity(self, hole_cards, board_cards, iterations=1000):
        """
        Calcula a equidade (probabilidade de vitória) usando simulação de Monte Carlo.
        """
        evaluator = Evaluator()
        win_count = 0
        
        for _ in range(iterations):
            deck = Deck()
            # Remove cartas conhecidas do deck
            known_cards = hole_cards + board_cards
            for card in known_cards:
                if card in deck.cards:
                    deck.cards.remove(card)
            
            deck.shuffle()
            
            # Simula mão do oponente e o restante da mesa
            opp_hand = deck.draw(2)
            remaining_count = 5 - len(board_cards)
            remaining_board = deck.draw(remaining_count) if remaining_count > 0 else []
            full_board = board_cards + remaining_board
            
            # Em treys, menor score é melhor (1 = Royal Flush)
            try:
                hero_score = evaluator.evaluate(hole_cards, full_board)
                opp_score = evaluator.evaluate(opp_hand, full_board)
                
                if hero_score < opp_score:
                    win_count += 1
                elif hero_score == opp_score:
                    win_count += 0.5
            except KeyError:
                continue # Ignora mãos problemáticas (raro, mas evita crash)
                
        return win_count / iterations

    def recommend_action(self, equity, pot_size, call_amount):
        """
        Sugere uma ação baseada na equidade e nas odds do pote.
        """
        # Pot Odds = Custo do Call / (Pote Atual + Custo do Call)
        if pot_size + call_amount == 0:
            pot_odds = 0
        else:
            pot_odds = call_amount / (pot_size + call_amount)
        
        recommendation = ""
        if equity > 0.75:
            recommendation = "ACE (ALL-IN / RAISE AGRESSIVO)"
        elif equity > pot_odds:
            recommendation = "CALL / CHECK"
        else:
            recommendation = "FOLD"
            
        return recommendation, pot_odds

    def analyze_screen(self):
        """
        Método principal para análise da tela, integrado com a interface GUI.
        Retorna uma string com os resultados da análise.
        """
        try:
            print("[CAPTURA] Iniciando captura de tela...")
            
            # Captura a tela usando método direto (mais confiável para GUI)
            img = self.capture_screen_direct()
            
            if img is None or img.size == 0:
                return "Erro: Não foi possível capturar a tela"
            
            print(f"[CAPTURA] Tela capturada com sucesso: {img.shape}")
            
            # Beep para indicar que a captura foi feita
            try:
                winsound.Beep(1000, 150)
            except:
                pass
            
            # Identifica as cartas
            print("[ANÁLISE] Identificando cartas...")
            hole, board = self.identify_cards(img, debug_mode=True)
            
            print(f"[ANÁLISE] Cartas encontradas - Mão: {len(hole)}, Mesa: {len(board)}")
            
            if len(hole) < 2:
                return "⚠️ Cartas insuficientes na mão (preciso de pelo menos 2)"
            
            # Calcula equidade
            print("[CÁLCULO] Calculando equidade...")
            equity = self.calculate_equity(hole, board)
            
            # Parâmetros do pote (valores padrão)
            pot = 100
            call = 20
            
            # Recomenda ação
            action, odds = self.recommend_action(equity, pot, call)
            
            # Formata resultado compacto
            result = f"🎯 ANÁLISE COMPLETA:\n"
            result += f"🂠 Suas cartas: {[Card.int_to_str(c) for c in hole]}\n"
            result += f"🃏 Mesa: {[Card.int_to_str(c) for c in board]}\n"
            result += f"📊 Equidade: {equity*100:.1f}% | Odds: {odds*100:.1f}%\n"
            result += f"💡 Ação: {action}"
            
            return result
            
        except Exception as e:
            return f"❌ Erro: {str(e)}"
    
    def capture_screen_direct(self):
        """
        Captura direta da tela usando PIL (mais confiável para GUI)
        """
        try:
            # Método direto usando PIL
            screenshot = ImageGrab.grab()
            img = np.array(screenshot)
            
            # Converter RGB para BGR (formato OpenCV)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            return img
            
        except Exception as e:
            print(f"[ERRO CAPTURA] Falha na captura direta: {e}")
            
            # Fallback para MSS
            try:
                if hasattr(self, 'sct'):
                    screenshot = self.sct.grab(self.sct.monitors[0])
                    img = np.array(screenshot)
                    
                    # Converter para BGR se necessário
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    elif len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    return img
            except Exception as e2:
                print(f"[ERRO CAPTURA] Fallback MSS também falhou: {e2}")
                
            return None

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if __name__ == "__main__":
    try:
        # Verifica permissão apenas para avisar, sem forçar reinício
        if not is_admin():
            print("\n" + "="*60)
            print("AVISO: O script não está rodando como Administrador.")
            print("Se a captura de tela do GG Poker ficar PRETA ou INVISÍVEL,")
            print("feche e abra este terminal como ADMINISTRADOR.")
            print("="*60 + "\n")
        
        analyzer = PokerAnalyzer()
        
        print("--- Iniciando Análise de Poker ---")
        print("Você tem 4 segundos para minimizar a IDE e mostrar a mesa de poker...")
        time.sleep(4)
        
        start_time = time.time()
        
        # 1. Captura da Tela
        print("\n[Info] Iniciando captura da tela...")
        img = analyzer.capture_screen() 
        # Beep para indicar que a captura foi feita (1000Hz, 200ms)
        winsound.Beep(1000, 200)
        
        # Salvar captura bruta para debug
        try:
            timestamp = int(time.time())
            debug_filename = f"debug_screenshot_raw_{timestamp}.png"
            print(f"[Debug] Salvando '{debug_filename}'...")
            cv2.imwrite(debug_filename, img)
            print(f"[Debug] Imagem {debug_filename} salva em: {os.getcwd()}")
        except Exception as e:
            print(f"[Erro] Falha ao salvar {debug_filename}: {e}")
        
        # 2. Identificação de Cartas
        try:
            hole, board = analyzer.identify_cards(img)
            print(f"Suas cartas: {[Card.int_to_str(c) for c in hole]}")
            print(f"Cartas na mesa: {[Card.int_to_str(c) for c in board]}")
        except Exception as e:
            print(f"[Erro] Ocorreu um problema na identificação de cartas: {e}")
            print("Tente verificar se as cartas estão visíveis na tela.")
            # Não sai mais, continua para permitir debug
            hole, board = [], []
        
        # 3. Cálculo de Equidade
        if len(hole) >= 2:
            print("Calculando equidade...")
            equity = analyzer.calculate_equity(hole, board)
        else:
            print("[Info] Cartas insuficientes para calcular equidade. (Preciso de pelo menos 2 na mão)")
            equity = 0
        
        # 4. Decisão
        pot = 100  # Exemplo: Pote de 100 fichas
        call = 20  # Exemplo: Precisa pagar 20 para continuar
        action, odds = analyzer.recommend_action(equity, pot, call)
        
        end_time = time.time()
        # Beep para indicar que o processamento foi concluído (1500Hz, 200ms)
        winsound.Beep(1500, 200)

        print("\n--- RESULTADO ---")
        print(f"Equidade Estimada: {equity*100:.2f}%")
        print(f"Pot Odds Necessárias: {odds*100:.2f}%")
        print(f"AÇÃO RECOMENDADA: {action}")
        print(f"Tempo de Processamento: {end_time - start_time:.4f} segundos")

    except Exception as e:
        print(f"\n[ERRO FATAL] O script falhou: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pass
