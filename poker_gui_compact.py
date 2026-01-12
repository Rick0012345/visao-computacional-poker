import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import os
import time
from datetime import datetime
from poker_analyzer import PokerAnalyzer

class PokerGUICompact:
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Analyzer")
        self.root.geometry("400x300")  # Tela bem menor
        self.root.resizable(False, False)  # Não permite redimensionar
        
        # Configurar o fechamento da janela
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variáveis de controle
        self.analyzer = None
        self.running = False
        self.thread = None
        self.analysis_count = 0
        
        # Cores modernas
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4CAF50"
        self.error_color = "#f44336"
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configurar estilo moderno
        self.root.configure(bg=self.bg_color)
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal compacto
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título compacto
        title_label = tk.Label(main_frame, text="🃏 POKER ANALYZER", 
                               bg=self.bg_color, fg=self.accent_color,
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Frame de status compacto
        status_frame = tk.Frame(main_frame, bg=self.bg_color)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicator
        self.status_label = tk.Label(status_frame, text="●", 
                                    bg=self.bg_color, fg="#4CAF50",
                                    font=("Arial", 16))
        self.status_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_text = tk.Label(status_frame, text="Pronto", 
                                   bg=self.bg_color, fg=self.fg_color,
                                   font=("Arial", 10))
        self.status_text.pack(side=tk.LEFT)
        
        # Contador de análises
        self.counter_label = tk.Label(status_frame, text="Análises: 0", 
                                     bg=self.bg_color, fg="#888888",
                                     font=("Arial", 9))
        self.counter_label.pack(side=tk.RIGHT)
        
        # Frame dos botões compacto
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=(0, 10))
        
        # Botão Analisar (principal)
        self.analyze_button = tk.Button(button_frame, text="▶ ANALISAR", 
                                       command=self.start_analysis,
                                       bg=self.accent_color, fg="white",
                                       font=("Arial", 10, "bold"),
                                       relief=tk.FLAT, padx=20, pady=8,
                                       cursor="hand2")
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Botão Parar
        self.stop_button = tk.Button(button_frame, text="⏹ PARAR", 
                                   command=self.stop_analysis,
                                   bg="#f44336", fg="white",
                                   font=("Arial", 10, "bold"),
                                   relief=tk.FLAT, padx=20, pady=8,
                                   cursor="hand2", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # Área de logs compacta
        log_frame = tk.LabelFrame(main_frame, text="Logs", 
                                 bg=self.bg_color, fg="#888888",
                                 font=("Arial", 9), padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=8, width=45,  # Bem menor
                               bg="#1e1e1e", fg="#00ff00",
                               font=("Courier", 8),  # Fonte menor
                               wrap=tk.WORD, padx=5, pady=5)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para logs
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Redirecionar stdout/stderr
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")
        
        # Log inicial
        self.log_message("🎯 Sistema pronto para análise")
        
    def log_message(self, message, message_type="info"):
        """Adiciona mensagem ao log com cores e timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Cores baseadas no tipo
        colors = {
            "info": "#00ff00",     # Verde
            "error": "#ff4444",    # Vermelho
            "warning": "#ffaa00", # Laranja
            "result": "#44aaff"    # Azul
        }
        
        color = colors.get(message_type, "#ffffff")
        
        self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.log_text.insert(tk.END, f"{message}\n", message_type)
        
        # Configurar tags de cores
        self.log_text.tag_config("timestamp", foreground="#888888")
        self.log_text.tag_config("info", foreground="#00ff00")
        self.log_text.tag_config("error", foreground="#ff4444")
        self.log_text.tag_config("warning", foreground="#ffaa00")
        self.log_text.tag_config("result", foreground="#44aaff")
        
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_analysis(self):
        """Inicia análise única (não contínua)"""
        if not self.running:
            self.running = True
            self.analysis_count += 1
            
            # Atualizar UI
            self.analyze_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(fg="#ffaa00")  # Amarelo = analisando
            self.status_text.config(text="Analisando...")
            self.counter_label.config(text=f"Análises: {self.analysis_count}")
            
            self.log_message("📸 Capturando tela...", "info")
            
            # Criar thread para análise
            self.thread = threading.Thread(target=self.run_single_analysis)
            self.thread.daemon = True
            self.thread.start()
            
    def stop_analysis(self):
        """Para a análise atual"""
        if self.running:
            self.running = False
            self.log_message("⏹ Análise interrompida", "warning")
            self.reset_ui()
            
    def run_single_analysis(self):
        """Executa uma única análise"""
        try:
            # Criar analisador se necessário
            if not hasattr(self, 'analyzer') or not self.analyzer:
                self.log_message("🔧 Inicializando analisador...", "info")
                try:
                    from poker_analyzer import PokerAnalyzer
                    self.analyzer = PokerAnalyzer()
                    self.log_message("✅ Analisador inicializado com sucesso", "info")
                except Exception as e:
                    self.log_message(f"❌ Erro ao inicializar analisador: {str(e)}", "error")
                    return
                
            # Executar análise com captura de tela melhorada
            self.log_message("📸 Capturando tela...", "info")
            
            # Capturar tela diretamente para garantir que funcione
            try:
                if not hasattr(self.analyzer, 'capture_screen_direct'):
                    self.log_message("❌ Método de captura não encontrado no analisador", "error")
                    return
                    
                img = self.analyzer.capture_screen_direct()
                if img is not None:
                    self.log_message(f"✅ Tela capturada com sucesso (formato: {img.shape})", "info")
                    
                    # Salvar imagem debug
                    try:
                        import cv2
                        import os
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_filename = f"debug_screenshot_{timestamp}.png"
                        cv2.imwrite(debug_filename, img)
                        self.log_message(f"📸 Imagem debug salva: {debug_filename}", "info")
                        self.log_message(f"📁 Caminho completo: {os.path.abspath(debug_filename)}", "info")
                    except Exception as e:
                        self.log_message(f"⚠️ Erro ao salvar imagem debug: {e}", "warning")
                    
                    # Verificar se a imagem tem dimensões válidas
                    if img.shape[0] > 0 and img.shape[1] > 0:
                        # Identificar cartas
                        self.log_message("🔍 Identificando cartas...", "info")
                        hole, board = self.analyzer.identify_cards(img, debug_mode=True)
                        
                        if len(hole) >= 2:
                            # Calcular equidade
                            self.log_message("📊 Calculando equidade...", "info")
                            equity = self.analyzer.calculate_equity(hole, board)
                            
                            # Recomendar ação
                            pot = 100
                            call = 20
                            action, odds = self.analyzer.recommend_action(equity, pot, call)
                            
                            # Mostrar resultados
                            self.log_message("🎯 ANÁLISE COMPLETA:", "result")
                            self.log_message(f"  🂠 Suas cartas: {[Card.int_to_str(c) for c in hole]}", "result")
                            self.log_message(f"  🃏 Mesa: {[Card.int_to_str(c) for c in board]}", "result")
                            self.log_message(f"  📊 Equidade: {equity*100:.1f}% | Odds: {odds*100:.1f}%", "result")
                            self.log_message(f"  💡 Ação: {action}", "result")
                            
                        else:
                            self.log_message("⚠️ Cartas insuficientes detectadas", "warning")
                    else:
                        self.log_message("❌ Imagem capturada tem dimensões inválidas", "error")
                        
                else:
                    self.log_message("❌ Falha na captura de tela - imagem retornou None", "error")
                    
            except Exception as capture_error:
                self.log_message(f"❌ Erro na captura: {str(capture_error)}", "error")
                self.log_message(f"📋 Tipo do erro: {type(capture_error).__name__}", "error")
                
                # Tentar método alternativo
                self.log_message("🔄 Tentando método alternativo...", "info")
                try:
                    if hasattr(self.analyzer, 'analyze_screen'):
                        result = self.analyzer.analyze_screen()
                        if result and "Erro" not in result:
                            self.log_message("🎯 ANÁLISE COMPLETA:", "result")
                            lines = result.split('\n')
                            for line in lines:
                                if any(key in line for key in ["Suas cartas:", "Cartas na mesa:", "Equidade:", "Ação:"]):
                                    self.log_message(f"  {line.strip()}", "result")
                        else:
                            self.log_message("❌ Nenhuma mesa detectada pelo método alternativo", "error")
                            if result:
                                self.log_message(f"📋 Resultado do método alternativo: {result}", "info")
                    else:
                        self.log_message("❌ Método alternativo não disponível", "error")
                except Exception as alt_error:
                    self.log_message(f"❌ Erro no método alternativo: {str(alt_error)}", "error")
                
        except Exception as e:
            self.log_message(f"❌ Erro geral na análise: {str(e)}", "error")
            self.log_message(f"📋 Tipo do erro geral: {type(e).__name__}", "error")
            
        finally:
            self.running = False
            self.root.after(500, self.reset_ui)
            
    def reset_ui(self):
        """Reseta UI para estado inicial"""
        self.analyze_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(fg="#4CAF50")  # Verde = pronto
        self.status_text.config(text="Pronto")
        
    def clear_logs(self):
        """Limpa os logs"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("🧹 Logs limpos", "info")
        
    def on_closing(self):
        """Lidar com fechamento"""
        if self.running:
            if messagebox.askokcancel("Confirmar", "Análise em andamento. Sair?"):
                self.running = False
                self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()


class TextRedirector:
    """Redireciona stdout/stderr para o widget Text"""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
        
    def write(self, str):
        # Filtrar mensagens desnecessárias
        filtered = str.strip()
        if filtered and not any(skip in filtered for skip in [
            "Debug]", "[Debug", "salvando", "debug_", "templates",
            "cards_templates", "unknown_cards"
        ]):
            self.widget.insert(tk.END, filtered + "\n", "info")
            self.widget.see(tk.END)
        
    def flush(self):
        pass


def main():
    root = tk.Tk()
    app = PokerGUICompact(root)
    root.mainloop()


if __name__ == "__main__":
    main()