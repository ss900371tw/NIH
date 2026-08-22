import os
import gc
import time
import torch
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from huggingface_hub import HfApi

HF_REPO_ID = "ss900371tw/medgemma-NIH-lora" 
HF_TOKEN = os.getenv("HF_TOKEN", "")

server_shared_model = None
server_shared_processor = None

status_lock = threading.Lock()

def load_server_model_once():
    global server_shared_model, server_shared_processor
    if server_shared_model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("❌ [Server 錯誤] Server 節點無可用 GPU，無法初始化 Unsloth 模型。")
            
        print("🚀 [Server] 檢測到 GPU，開始載入 Unsloth MedGemma...")
        from .task import create_model_and_tokenizer
        server_shared_model, server_shared_processor = create_model_and_tokenizer()
        
    return server_shared_model, server_shared_processor

class ServerDashboard(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self._status = "IDLE"  
        self._current_round = 0  
        self._connected_clients = [] 
        self.root = None
        self.canvas = None

    @property
    def status(self):
        with status_lock: return self._status
    @status.setter
    def status(self, val):
        with status_lock: self._status = val

    @property
    def current_round(self):
        with status_lock: return self._current_round
    @current_round.setter
    def current_round(self, val):
        with status_lock: self._current_round = val

    @property
    def connected_clients(self):
        with status_lock: return list(self._connected_clients)
    @connected_clients.setter
    def connected_clients(self, val):
        with status_lock: self._connected_clients = val

    def draw_status_light(self):
        if not self.canvas:
            return
        self.canvas.delete("bulb") 
        
        current_status = self.status
        if current_status == "AGGREGATING":
            c_base = "#D4AF37"   
            c_light = "#FFD700"  
            c_text = "black"
            status_text = "AGGREGATING"
        elif current_status == "PUSHING_TO_HF":
            c_base = "#2980b9"   
            c_light = "#3498db"  
            c_text = "white"
            status_text = "PUSH TO HUB"
        elif current_status == "TRAINING":
            c_base = "#27ae60"
            c_light = "#2ecc71"
            c_text = "white"
            status_text = "CLIENT TRAINING"
        else:
            c_base = "#444444"   
            c_light = "#222222"  
            c_text = "#666666"
            status_text = "SERVER IDLE"

        self.canvas.create_oval(125, 35, 275, 185, fill=c_base, outline="", tags="bulb")
        self.canvas.create_oval(130, 40, 270, 180, fill=c_light, outline="", tags="bulb")
        
        highlight_color = "#ffffff" if current_status in ["AGGREGATING", "PUSHING_TO_HF", "TRAINING"] else "#333333"
        self.canvas.create_oval(170, 50, 230, 75, fill=highlight_color, outline="", tags="bulb")

        threads = [(165, 175, 235, 190), (170, 187, 230, 200), (175, 197, 225, 210)]
        for (x1, y1, x2, y2) in threads:
            self.canvas.create_oval(x1, y1, x2, y2, fill="#bdc3c7", outline="#95a5a6", tags="bulb")
        
        self.canvas.create_oval(185, 205, 215, 220, fill="#34495e", outline="", tags="bulb")

        r_num = self.current_round
        round_text = f"★ GLOBAL ROUND: {r_num} ★" if r_num > 0 else "★ SERVER INITIALIZING ★"
        self.canvas.create_text(200, 18, text=round_text, fill="#00FF00", font=("Consolas", 12, "bold"), tags="bulb")
        self.canvas.create_text(200, 110, text=status_text, fill=c_text, font=("Microsoft JhengHei", 12, "bold"), tags="bulb")

    def run(self):
        self.root = tk.Tk()
        self.root.title("🌸 Flower Server Dashboard")
        self.root.geometry("400x650")
        self.root.configure(bg="#121212")
        self.canvas = tk.Canvas(self.root, width=400, height=210, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(pady=10)
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=20, pady=5)
        title_label = tk.Label(self.root, text="Connected Clients List", bg="#121212", fg="#00FF00", font=("Consolas", 14, "bold"))
        title_label.pack()
        self.client_display = tk.Text(self.root, bg="#000000", fg="#00FF00", font=("Consolas", 10), height=15, width=42)
        self.client_display.pack(pady=10, padx=20)
        self.count_label = tk.Label(self.root, text="Total: 0 clients", bg="#121212", fg="white")
        self.count_label.pack()
        self.update_ui()
        self.root.mainloop()

    def update_ui(self):
        if not self.root or not tk._default_root:
            return
        try:
            self.draw_status_light()
            self.client_display.config(state=tk.NORMAL)
            self.client_display.delete('1.0', tk.END)
            clients = self.connected_clients
            if not clients:
                self.client_display.insert(tk.END, "> [System] Waiting for client connections...\n")
            else:
                for idx, cid in enumerate(clients):
                    self.client_display.insert(tk.END, f"  ● [{idx+1}] ID: {cid}\n")
            self.client_display.config(state=tk.DISABLED)
            self.count_label.config(text=f"Total Connected Clients: {len(clients)}")
            self.root.after(500, self.update_ui)
        except Exception:
            pass

monitor = None

def start_server_monitor_safely():
    global monitor
    if monitor is None or not monitor.is_alive():
        monitor = ServerDashboard()
        monitor.start()
        time.sleep(0.2)

class CustomFedAvg(FedAvg):
    def __init__(self, total_rounds, *args, **kwargs):
        self.latest_parameters = None 
        self.total_rounds = total_rounds  
        self.best_eval_loss = float("inf")
        self.best_parameters = None
        super().__init__(*args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        start_server_monitor_safely()
        clients_dict = client_manager.all()
        if monitor:
            monitor.current_round = server_round  
            monitor.connected_clients = list(clients_dict.keys())
            monitor.status = "TRAINING"
            
        if server_round > 1 and self.latest_parameters is not None:
            parameters = self.latest_parameters
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures):
        start_server_monitor_safely()
        if monitor:
            monitor.current_round = server_round
            monitor.status = "AGGREGATING"
            
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is None:
            return None
            
        aggregated_parameters, _ = aggregated_result
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            weights = parameters_to_ndarrays(aggregated_parameters)
            
            os.makedirs("SERVER", exist_ok=True)
            local_pth_path = os.path.join("SERVER", f"lora_round_{server_round}.pth")
            try:
                torch.save(weights, local_pth_path)
            except Exception as e:
                print(f"⚠️ 本地備份失敗: {e}")

        if monitor:
            monitor.status = "IDLE"
        return aggregated_result

    def configure_evaluate(self, server_round, parameters, client_manager):
        start_server_monitor_safely()
        if monitor:
            monitor.current_round = server_round
        if self.latest_parameters is not None:
            parameters = self.latest_parameters
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_result is None:
            return None
            
        current_loss, metrics = aggregated_result
        
        if current_loss is not None:
            print(f"📊 [Round {server_round} 評估結果] 當前全球 Loss: {current_loss:.4f} | 歷史最佳 Loss: {self.best_eval_loss:.4f}")
            
            if current_loss <= self.best_eval_loss:
                self.best_eval_loss = current_loss
                self.best_parameters = self.latest_parameters
                
                print(f"🏆 檢測到更好的模型！更新 Best Model (Round {server_round})...")
                
                if self.best_parameters is not None:
                    best_weights = parameters_to_ndarrays(self.best_parameters)
                    
                    os.makedirs("SERVER", exist_ok=True)
                    best_local_path = os.path.join("SERVER", "lora_best_model.pth")
                    try:
                        torch.save(best_weights, best_local_path)
                        print(f"💾 本地最佳模型已更新儲存至: {best_local_path}")
                    except Exception as e:
                        print(f"⚠️ 本地儲存最佳模型失敗: {e}")
                    
                    if monitor:
                        monitor.status = "PUSHING_TO_HF" 
                        
                    server_model = None
                    try:
                        if not torch.cuda.is_available():
                            raise RuntimeError("❌ [CUDA 錯誤] Server 端未偵測到 GPU，無法進行 Unsloth 模型上傳！")

                        print(f"🚀 [GPU 模式上傳] 正在透過 Unsloth 將完整 LoRA 結構推送到 {HF_REPO_ID}...")
                        server_model, server_processor = load_server_model_once()
                        
                        from .task import set_weights
                        set_weights(server_model, best_weights)
                        
                        server_model.push_to_hub(repo_id=HF_REPO_ID, token=HF_TOKEN, private=True)
                        if server_round == 1 and hasattr(server_processor, "push_to_hub"):
                            server_processor.push_to_hub(repo_id=HF_REPO_ID, token=HF_TOKEN)
                        print(f"✅ [Hub Best 更新] GPU Unsloth 結構上傳成功！")
                            
                    except Exception as e:
                        print(f"❌ 權重上傳 Hugging Face 失敗: {e}")
                            
                    finally:
                        if server_model is not None:
                            del server_model
                            server_model = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        if monitor:
                            monitor.status = "IDLE"
                                            
        return aggregated_result

def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    if not examples or sum(examples) == 0:
        return {"loss": 0.0}
    return {"loss": sum(num * m.get("loss", 0) for num, m in metrics) / sum(examples)}

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    print("⚡ [Server 初始化] 正在載入全球初始化 LoRA 權重結構...")
    
    from .task import get_weights
    server_model, _ = load_server_model_once()
    init_weights = get_weights(server_model)
    init_parameters = ndarrays_to_parameters(init_weights)

    strategy = CustomFedAvg(
        total_rounds=num_rounds,
        fraction_fit=1.0,           
        fraction_evaluate=1.0,      
        min_available_clients=2,    
        min_fit_clients=2,          
        min_evaluate_clients=2,     
        initial_parameters=init_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda sr: {"server_round": sr},
        on_evaluate_config_fn=lambda sr: {"server_round": sr},
    )

    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)
