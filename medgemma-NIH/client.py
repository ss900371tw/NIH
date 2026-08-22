import os
import gc
import time
import socket
import threading
import fcntl
import torch
import tkinter as tk
import numpy as np
import nvflare.client as flare
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.cuda.init()
    
sip = "100.116.129.103"

def acquire_gpu_lock():
    lock_file = "/tmp/flower_gpu.lock"
    f = open(lock_file, 'w')
    fcntl.flock(f, fcntl.LOCK_EX)
    return f

def release_gpu_lock(f):
    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()

class ConnectionMonitor(threading.Thread):
    def __init__(self, server_ip=sip):
        super().__init__()
        self.server_ip = server_ip
        self.daemon = True
        self.blink_state = False
        self.status = "IDLE"  
        self.is_connected = True
        self.current_round = 0  
        self.root = None
        self.canvas = None

    def update_status(self, status, connected=True, current_round=None):
        self.status = status
        self.is_connected = connected
        if current_round is not None:
            self.current_round = current_round

    def draw_crystal_ball(self, color_base, color_light):
        self.canvas.delete("ball") 
        self.canvas.create_oval(50, 50, 350, 350, fill="#1a1a1a", outline="", tags="ball")
        self.canvas.create_oval(60, 60, 340, 340, fill=color_base, outline="", tags="ball")
        self.canvas.create_oval(80, 80, 320, 320, fill=color_light, outline="", tags="ball")
        self.canvas.create_oval(140, 90, 240, 130, fill="#ffffff", outline="", tags="ball")       
        self.canvas.create_oval(170, 375, 230, 405, fill="#34495e", outline="", tags="ball")
        threads = [(125, 320, 275, 340), (130, 335, 270, 355), (140, 350, 260, 370), (150, 365, 250, 385)]
        for (x1, y1, x2, y2) in threads:
            self.canvas.create_oval(x1, y1, x2, y2, fill="#95a5a6", outline="#7f8c8d", width=1, tags="ball")

    def run(self):
        try:
            self.root = tk.Tk()
            self.root.title("FL Client Monitor")
            self.root.attributes("-topmost", True)
            self.root.geometry("400x500")      
            self.canvas = tk.Canvas(self.root, width=400, height=500, highlightthickness=0)
            self.canvas.pack()
            
            def on_closing():
                try:
                    self.root.quit()
                    self.root.destroy()
                except:
                    pass
            self.root.protocol("WM_DELETE_WINDOW", on_closing)
            
            self.update_ui()
            self.root.mainloop()
        except Exception:
            pass

    def update_ui(self):
        if not self.root or not tk._default_root:
            return

        is_connected = self.is_connected
        self.blink_state = not self.blink_state
        
        if not is_connected:
            c_base = "#7b0000" if self.blink_state else "#b71c1c"
            c_light = "#c0392b" if self.blink_state else "#ef5350"
            message = "DISCONNECTED"
            bg_color = "#2a0d12"
        elif self.status == "TRAINING":
            c_base = "#1b5e20" if self.blink_state else "#2e7d32"
            c_light = "#4caf50" if self.blink_state else "#81c784"
            message = "TRAINING"
            bg_color = "#0d1b2a"
        elif self.status == "EVALUATING":
            c_base = "#0d47a1" if self.blink_state else "#1565c0"
            c_light = "#2196f3" if self.blink_state else "#64b5f6"
            message = "EVALUATING"
            bg_color = "#0d1b2a"
        else:
            c_base = "#455a64"
            c_light = "#78909c"
            message = "IDLE"
            bg_color = "#121212"

        try:
            self.root.configure(bg=bg_color)
            self.canvas.configure(bg=bg_color)
            self.draw_crystal_ball(c_base, c_light)
            
            self.canvas.delete("text")
            round_text = f"——  ROUND {self.current_round}  ——" if self.current_round > 0 else "——  WAITING  ——"
            self.canvas.create_text(200, 25, text=round_text, fill="#00FF00", font=("Consolas", 14, "bold"), tags="text")
            self.canvas.create_text(200, 200, text=message, fill="white", font=("Microsoft JhengHei", 26, "bold"), tags="text")
            self.canvas.create_text(200, 460, text=f"Target: {self.server_ip}", fill="#bdc3c7", font=("Consolas", 10), tags="text")
            
            self.root.after(800, self.update_ui)
        except Exception:
            pass

monitor_thread = None
shared_model = None
shared_processor = None
shared_train_dataset = None
shared_test_dataset = None

SITE_PATHS_MAPPING = {
    "site-1": "/opt/toolkit/NIH/site-1/",
    "site-2": "/opt/toolkit/NIH/site-2/",
    "site-3": "/opt/toolkit/NIH/site-3/",
    "site-4": "/opt/toolkit/NIH/site-4/",
}

current_cached_site = None 

# 🌟 新增：專門依據當前 Round 動態生成 Training Dataset
def update_train_dataset_for_round(site_name, current_round):
    global shared_train_dataset
    from .task import prepare_vqa_dataset
    
    base_dir = SITE_PATHS_MAPPING.get(site_name, f"/opt/toolkit/NIH/{site_name}/")
    train_csv_path = os.path.join(base_dir, "train.csv")
    train_images_dir = os.path.join(base_dir, "images")
    
    # 動態 Seed 計算：每增加一輪，Seed +1 (例如 Round 1: 3405, Round 2: 3406...)
    dynamic_seed = 3405 + (current_round - 1)
    print(f"🔄 [{site_name}] 正在為 Round {current_round} 生成動態訓練集 (Seed: {dynamic_seed})...")
    
    shared_train_dataset = prepare_vqa_dataset(
        train_csv_path, 
        train_images_dir, 
        3000, 
        is_train=True, 
        current_seed=dynamic_seed
    )

def load_resources_once(site_name="site-1"):
    global shared_model, shared_processor, shared_test_dataset, current_cached_site
    
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        raise RuntimeError(
            f"❌ [{site_name}] 當前系統未檢測到可用 GPU (CUDA_VISIBLE_DEVICES={cuda_env})！"
            "此專案已限定只能使用 GPU 執行，禁止 CPU 降級運算。"
        )

    if shared_model is None:
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"🚀 [{site_name} - GPU 專用模式] (CUDA_VISIBLE_DEVICES={cuda_env}) 初始化 MedGemma + Unsloth 加速核心...")
        try:
            from .task import create_model_and_tokenizer
            shared_model, shared_processor = create_model_and_tokenizer()
        except Exception as gpu_err:
            print(f"❌ [{site_name}] GPU 模型初始化失敗: {gpu_err}")
            raise gpu_err

    # 🌟 測試集維持只載入與快取一次（評估基準固定）
    if shared_test_dataset is None or current_cached_site != site_name:
        print(f"📂 [{site_name}] 正在載入該站點專屬的測試數據集 (固定 Seed)...")
        from .task import prepare_vqa_dataset
        
        base_dir = SITE_PATHS_MAPPING.get(site_name, f"/opt/toolkit/NIH/{site_name}/")
        test_csv_path = os.path.join(base_dir, "test.csv")
        test_images_dir = os.path.join(base_dir, "images")
        
        shared_test_dataset = prepare_vqa_dataset(test_csv_path, test_images_dir, 500, is_train=False)
        current_cached_site = site_name

def start_client_monitor_safely(current_round, initial_status="TRAINING"):
    global monitor_thread
    if monitor_thread is None or not monitor_thread.is_alive():
        print("🖥️ [Monitor] 正在啟動 Tkinter 聯邦學習水晶球監控面板...")
        monitor_thread = ConnectionMonitor(server_ip=sip)
        monitor_thread.status = initial_status
        monitor_thread.current_round = current_round
        monitor_thread.start()
        time.sleep(0.1)

def clear_global_model():
    global shared_model, shared_processor
    
    if shared_model is not None:
        try:
            shared_model.zero_grad(set_to_none=True)
        except Exception:
            pass
            
        del shared_model
        del shared_processor
        shared_model = None
        shared_processor = None
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        print("🧹 [GPU Memory] 已成功釋放 VRAM 並完成 CUDA 快取同步。")

class FlowerClient(NumPyClient):
    def __init__(self, context: Context):
        super().__init__()
        self.flwr_context = context
        partition_id = self.flwr_context.node_config.get("partition-id", 0)
        self.site_name = f"site-{partition_id + 1}"

    def get_parameters(self, config):
        print(f"🤝 [{self.site_name}] 取得本地端權重結構...")
        load_resources_once(self.site_name)
        from .task import get_weights
        weights = get_weights(shared_model)
        clear_global_model()
        return weights

    def fit(self, parameters, config):
        global shared_model, shared_processor, shared_train_dataset
        current_round = config.get("server_round", 1) 
        start_client_monitor_safely(current_round, "TRAINING")

        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.update_status("TRAINING", connected=True, current_round=current_round)

        worker_output = {}

        def async_train_worker():
            gpu_lock = acquire_gpu_lock()
            try:
                print(f"📦 [{self.site_name}] 收到來自 Server 的 Round {current_round} 訓練指令...")
                load_resources_once(self.site_name)
                
                # 🌟 關鍵修訂：每回合開始微調前，呼叫動態重新抽樣訓練集
                update_train_dataset_for_round(self.site_name, current_round)
                
                from .task import set_weights, train_local, get_weights
                set_weights(shared_model, parameters)

                results = train_local(
                    model=shared_model,
                    processor=shared_processor,
                    train_dataset=shared_train_dataset,
                    epochs=1
                )
                worker_output["weights"] = get_weights(shared_model)
                worker_output["results"] = results
            except Exception as e:
                worker_output["exception"] = e
            finally:
                release_gpu_lock(gpu_lock)

        train_thread = threading.Thread(target=async_train_worker)
        train_thread.start()

        while train_thread.is_alive():
            time.sleep(1.0)

        if "exception" in worker_output:
            clear_global_model()
            raise worker_output["exception"]

        weights = worker_output.get("weights")
        results = worker_output.get("results")

        if weights is None:
            weights = parameters

        formatted_metrics = {}
        if results is not None:
            raw_metrics = results.metrics if hasattr(results, "metrics") else (results if isinstance(results, dict) else {})
            for k, v in raw_metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    formatted_metrics[k] = v
                elif hasattr(v, "item"): 
                    formatted_metrics[k] = v.item()

        train_size = len(shared_train_dataset) if shared_train_dataset is not None else 0

        if train_size == 0:
            clear_global_model()
            raise RuntimeError(f"❌ [{self.site_name}] 本地訓練數據集中沒有任何可用的圖片/樣本！")

        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.update_status("IDLE", connected=True, current_round=current_round)

        clear_global_model()
        
        return weights, train_size, formatted_metrics

    def evaluate(self, parameters, config):
        global shared_model, shared_processor, shared_test_dataset
        current_round = config.get("server_round", 1)
        start_client_monitor_safely(current_round, "EVALUATING")
        
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.update_status("EVALUATING", connected=True, current_round=current_round)

        worker_output = {}

        def async_eval_worker():
            gpu_lock = acquire_gpu_lock()
            try:
                load_resources_once(self.site_name)
                from .task import set_weights, evaluate_local
                set_weights(shared_model, parameters)
                    
                loss, metrics = evaluate_local(shared_model, shared_processor, shared_test_dataset)
                worker_output["loss"] = loss
                worker_output["metrics"] = metrics
            except Exception as e:
                worker_output["exception"] = e
            finally:
                release_gpu_lock(gpu_lock)

        eval_thread = threading.Thread(target=async_eval_worker)
        eval_thread.start()

        while eval_thread.is_alive():
            time.sleep(1.0)

        if "exception" in worker_output:
            clear_global_model()
            raise worker_output["exception"]

        loss = worker_output.get("loss")
        metrics = worker_output.get("metrics")
        test_size = len(shared_test_dataset) if shared_test_dataset is not None else 0

        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.update_status("IDLE", connected=True, current_round=current_round)
                
        clear_global_model()

        return float(loss), test_size, metrics

def client_fn(context: Context):
    return FlowerClient(context).to_client()

app = ClientApp(client_fn=client_fn)

@app.lifespan()
def lifespan(ctxt: Context) -> None:
    try:
        flare.init()
    except Exception as e:
        print(f"⚠️ NVFlare init skipped: {e}")
    yield
    try:
        flare.shutdown()
    except:
        pass
