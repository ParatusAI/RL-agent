# for_ryan_file_watcher.py - File-based integration for Ryan's RL agent

import os
import time
import csv
import threading
from datetime import datetime
from typing import Dict, Any, Callable
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IsaiahPredictionWatcher:
    """
    File watcher that monitors Isaiah's prediction CSV files
    Automatically notifies Ryan's RL agent when new predictions arrive
    """
    
    def __init__(self, predictions_folder: str = "real_time_predictions"):
        self.predictions_folder = predictions_folder
        self.is_watching = False
        self.processed_files = set()
        self.prediction_callback = None
        self.watch_thread = None
        
        # Create folder if it doesn't exist
        os.makedirs(self.predictions_folder, exist_ok=True)
        
        logger.info(f"🔍 Prediction watcher initialized")
        logger.info(f"📁 Monitoring folder: {self.predictions_folder}")
    
    def set_prediction_callback(self, callback: Callable[[Dict[str, Any], str], None]):
        """
        Set callback function that gets called when new prediction arrives
        
        Args:
            callback: Function that takes (prediction_dict, filename) as arguments
        """
        self.prediction_callback = callback
        logger.info("✅ Prediction callback registered")
    
    def start_watching(self):
        """Start watching for new prediction files"""
        if self.is_watching:
            logger.warning("⚠️  Already watching for predictions!")
            return
        
        if not self.prediction_callback:
            logger.error("❌ No callback set! Use set_prediction_callback() first")
            return
        
        self.is_watching = True
        
        # Get existing files to avoid processing them again
        existing_files = {f for f in os.listdir(self.predictions_folder) if f.endswith('.csv')}
        self.processed_files.update(existing_files)
        
        logger.info(f"🚀 Starting prediction file watcher...")
        logger.info(f"📋 Ignoring {len(existing_files)} existing files")
        
        # Start watching in separate thread
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        
        return self.watch_thread
    
    def stop_watching(self):
        """Stop watching for prediction files"""
        self.is_watching = False
        logger.info("🛑 Stopped watching for predictions")
    
    def _watch_loop(self):
        """Main watching loop - runs in separate thread"""
        logger.info("👀 File watcher started - waiting for new predictions...")
        
        while self.is_watching:
            try:
                # Check for new CSV files
                current_files = {f for f in os.listdir(self.predictions_folder) if f.endswith('.csv')}
                new_files = current_files - self.processed_files
                
                # Process new files
                for filename in sorted(new_files):
                    if self.is_watching:  # Check if still watching
                        self._process_new_prediction_file(filename)
                        self.processed_files.add(filename)
                
                # Wait before checking again
                time.sleep(1)  # Check every 1 second
                
            except Exception as e:
                logger.error(f"❌ Error in watch loop: {e}")
                time.sleep(2)
    
    def _process_new_prediction_file(self, filename: str):
        """Process a new prediction CSV file"""
        try:
            file_path = os.path.join(self.predictions_folder, filename)
            
            # Read CSV file with key-value pairs
            prediction_dict = self._read_prediction_csv(file_path)
            
            if prediction_dict:
                # Extract timestamp from filename (e.g., prediction_T030s.csv)
                timestamp_part = filename.replace('prediction_', '').replace('.csv', '')
                
                logger.info(f"📥 NEW PREDICTION: {filename}")
                logger.info(f"   PLQY: {prediction_dict.get('predicted_plqy', 'N/A')}")
                logger.info(f"   Peak: {prediction_dict.get('predicted_emission_peak', 'N/A')} nm")
                logger.info(f"   FWHM: {prediction_dict.get('predicted_fwhm', 'N/A')} nm")
                
                # Call Ryan's RL callback
                if self.prediction_callback:
                    self.prediction_callback(prediction_dict, filename)
                
            else:
                logger.warning(f"⚠️  Failed to read prediction from {filename}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
    
    def _read_prediction_csv(self, file_path: str) -> Dict[str, Any]:
        """Read prediction data from CSV file with key-value pairs"""
        try:
            prediction_dict = {}
            
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Skip header row if present
                for row in rows[1:]:  # Skip first row (header)
                    if len(row) >= 2:
                        key = row[0].strip()
                        value = row[1].strip()
                        
                        # Convert to appropriate type
                        if key in ['predicted_plqy', 'predicted_emission_peak', 'predicted_fwhm', 'confidence']:
                            try:
                                prediction_dict[key] = float(value)
                            except ValueError:
                                prediction_dict[key] = value
                        else:
                            prediction_dict[key] = value
            
            return prediction_dict
            
        except Exception as e:
            logger.error(f"❌ Error reading CSV {file_path}: {e}")
            return {}

class RyanRLEnvironment:
    """
    Example RL Environment that uses file watching for Isaiah's predictions
    """
    
    def __init__(self):
        self.prediction_watcher = IsaiahPredictionWatcher()
        self.latest_prediction = None
        self.prediction_history = []
        
        # Set up callback for when predictions arrive
        self.prediction_watcher.set_prediction_callback(self._on_new_prediction)
        
        # Current synthesis parameters
        self.current_cs_flow = 1.0
        self.current_pb_flow = 1.0
        self.current_temperature = 80.0
        self.current_residence_time = 120.0
        
        logger.info("🤖 Ryan's RL Environment initialized")
    
    def start_monitoring(self):
        """Start monitoring for Isaiah's predictions"""
        logger.info("🔍 Starting prediction monitoring...")
        self.prediction_watcher.start_watching()
    
    def stop_monitoring(self):
        """Stop monitoring predictions"""
        self.prediction_watcher.stop_watching()
    
    def _on_new_prediction(self, prediction: Dict[str, Any], filename: str):
        """
        Callback function called when Isaiah's CNN makes a new prediction
        This is where Ryan's RL agent makes decisions!
        """
        logger.info(f"🤖 RYAN'S RL AGENT ACTIVATED!")
        
        # Store the prediction
        self.latest_prediction = prediction
        self.prediction_history.append({
            'filename': filename,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
        # Make RL decision based on prediction
        decision = self._make_rl_decision(prediction)
        
        # Take action based on decision
        self._execute_rl_action(decision)
    
    def _make_rl_decision(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ryan's RL decision making logic
        """
        plqy = prediction.get('predicted_plqy', 0.0)
        emission_peak = prediction.get('predicted_emission_peak', 515.0)
        fwhm = prediction.get('predicted_fwhm', 25.0)
        confidence = prediction.get('confidence', 0.0)
        
        logger.info(f"🧠 RL DECISION MAKING:")
        logger.info(f"   Analyzing: PLQY={plqy:.3f}, Peak={emission_peak:.1f}nm, FWHM={fwhm:.1f}nm")
        
        # RL decision logic
        if plqy > 0.8 and 515 <= emission_peak <= 525:
            decision = {
                'action': 'CONTINUE',
                'reason': 'Target properties achieved',
                'parameter_changes': {},
                'reward': self._calculate_reward(prediction)
            }
        elif plqy > 0.6:
            # Need optimization
            changes = {}
            
            if emission_peak < 515:
                changes['temperature'] = self.current_temperature + 5  # Red-shift
                reason = 'Increasing temperature to red-shift emission'
            elif emission_peak > 525:
                changes['temperature'] = self.current_temperature - 5  # Blue-shift
                reason = 'Decreasing temperature to blue-shift emission'
            else:
                changes['cs_flow'] = self.current_cs_flow * 1.1  # Improve PLQY
                reason = 'Increasing Cs flow to improve PLQY'
            
            decision = {
                'action': 'OPTIMIZE',
                'reason': reason,
                'parameter_changes': changes,
                'reward': self._calculate_reward(prediction)
            }
        else:
            # Poor quality - major changes needed
            decision = {
                'action': 'MAJOR_CORRECTION',
                'reason': 'Poor quality detected - major parameter adjustment',
                'parameter_changes': {
                    'cs_flow': self.current_cs_flow * 0.8,
                    'pb_flow': self.current_pb_flow * 1.2,
                    'temperature': 75.0
                },
                'reward': self._calculate_reward(prediction)
            }
        
        logger.info(f"✅ RL DECISION: {decision['action']}")
        logger.info(f"   Reason: {decision['reason']}")
        logger.info(f"   Reward: {decision['reward']:.3f}")
        
        return decision
    
    def _calculate_reward(self, prediction: Dict[str, Any]) -> float:
        """Calculate RL reward from prediction"""
        plqy = prediction.get('predicted_plqy', 0.0)
        emission_peak = prediction.get('predicted_emission_peak', 515.0)
        confidence = prediction.get('confidence', 0.0)
        
        # Reward function
        plqy_reward = plqy
        peak_penalty = abs(emission_peak - 520) / 20  # Penalty for deviation from 520nm
        confidence_bonus = confidence * 0.1
        
        total_reward = plqy_reward - peak_penalty + confidence_bonus
        return total_reward
    
    def _execute_rl_action(self, decision: Dict[str, Any]):
        """Execute the RL decision"""
        action = decision['action']
        changes = decision.get('parameter_changes', {})
        
        if action in ['OPTIMIZE', 'MAJOR_CORRECTION'] and changes:
            logger.info(f"⚙️  UPDATING SYNTHESIS PARAMETERS:")
            
            # Update parameters
            for param, new_value in changes.items():
                old_value = getattr(self, f'current_{param}')
                setattr(self, f'current_{param}', new_value)
                logger.info(f"   {param}: {old_value:.1f} → {new_value:.1f}")
            
            # In real system, this would send to Aroyston's hardware
            logger.info(f"📡 Sending updated parameters to Aroyston's hardware...")
            
        elif action == 'CONTINUE':
            logger.info(f"✅ MAINTAINING current parameters - synthesis on target!")
        
        logger.info(f"🔄 Waiting for next spectral measurement...")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current RL environment status"""
        return {
            'monitoring': self.prediction_watcher.is_watching,
            'predictions_received': len(self.prediction_history),
            'latest_prediction': self.latest_prediction,
            'current_parameters': {
                'cs_flow': self.current_cs_flow,
                'pb_flow': self.current_pb_flow,
                'temperature': self.current_temperature,
                'residence_time': self.current_residence_time
            }
        }

# Demo script for Ryan
def demo_file_watcher_integration():
    """Demo showing how Ryan's RL integrates with Isaiah's file system"""
    
    print("🤖 Ryan's RL Agent - File Watcher Integration Demo")
    print("=" * 60)
    print("🔍 This demo shows how Ryan's RL receives Isaiah's predictions")
    print("📁 Monitoring folder: real_time_predictions/")
    print("⏰ Will respond immediately when new CSV files appear")
    print("=" * 60)
    
    # Initialize Ryan's RL environment
    rl_env = RyanRLEnvironment()
    
    # Start monitoring
    rl_env.start_monitoring()
    
    print("\n👀 MONITORING STARTED - Waiting for Isaiah's predictions...")
    print("💡 Run Isaiah's MVP demo in another terminal to see this in action!")
    print("🛑 Press Ctrl+C to stop\n")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(5)
            
            # Show status every 30 seconds
            status = rl_env.get_status()
            if status['predictions_received'] > 0:
                print(f"📊 Status: {status['predictions_received']} predictions received")
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        rl_env.stop_monitoring()
    
    # Show final summary
    status = rl_env.get_status()
    print(f"\n📊 Final Status:")
    print(f"   Predictions processed: {status['predictions_received']}")
    if status['latest_prediction']:
        latest = status['latest_prediction']
        print(f"   Last PLQY: {latest.get('predicted_plqy', 'N/A')}")
        print(f"   Last Peak: {latest.get('predicted_emission_peak', 'N/A')} nm")
    
    print(f"\n✅ File watcher integration demo completed!")

if __name__ == "__main__":
    demo_file_watcher_integration()