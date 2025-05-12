#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NQ Alpha Elite v7.0.0 - Advanced Deployment Script
Deploy the high-frequency trading system for Nasdaq 100 E-mini futures
"""

import os
import sys
import argparse
import json
import logging
import time
import signal
import threading
from datetime import datetime
import traceback
import importlib
import yaml

# Setup path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def setup_logging(log_level='INFO', log_dir='logs'):
    """Setup logging configuration
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'nq_elite_{timestamp}.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Configure formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    
    # Return logger
    return logging.getLogger('NQElite.Deployer')

def load_config(config_path):
    """Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            print(f"Error: Configuration file not found: {config_path}")
            return None
        
        # Load configuration based on file extension
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            print(f"Error: Unsupported configuration file format: {config_path}")
            return None
        
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return None

def create_default_config():
    """Create default configuration
    
    Returns:
        dict: Default configuration
    """
    default_config = {
        "system": {
            "name": "NQ Alpha Elite",
            "version": "7.0.0",
            "description": "High-Frequency Trading System for Nasdaq 100 E-mini Futures",
            "log_level": "INFO",
            "log_dir": "logs",
            "data_dir": "data",
            "models_dir": "models",
            "debug_mode": False
        },
        "strategy": {
            "name": "NQEliteStrategy",
            "timeframes": ["10s", "1m", "5m", "15m", "1h", "4h", "daily"],
            "primary_timeframe": "1m",
            "signal_timeframe": "1m",
            "entry_threshold": 0.6,
            "exit_threshold": 0.4,
            "entry_confidence_threshold": 0.65,
            "exit_confidence_threshold": 0.55,
            "use_microstructure_features": True,
            "use_multi_timeframe": True,
            "reversal_detection": True,
            "regime_adaptation": True,
            "feature_importances": True,
            "momentum_confirmation": True,
            "trend_following": True,
            "mean_reversion": False,
            "use_ensemble": True,
            "model_params": {
                "model_type": "transformer_hf",
                "model_path": "models/nq_transformer_hf_model.h5",
                "ensemble_models": ["transformer_hf", "lstm_cnn", "gradient_boost"],
                "voting_method": "weighted",
                "model_weights": [0.6, 0.3, 0.1]
            }
        },
        "data": {
            "source": "combined",
            "providers": ["internal", "external"],
            "use_cache": True,
            "update_frequency": {
                "10s": 10,
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "1h": 3600,
                "4h": 14400,
                "daily": 86400
            },
            "cache_duration": {
                "10s": 3600,
                "1m": 86400,
                "5m": 604800,
                "15m": 2592000,
                "1h": 7776000,
                "4h": 15552000,
                "daily": 31536000
            },
            "features": {
                "technical": True,
                "microstructure": True,
                "sentiment": True,
                "fundamental": False,
                "alternative": True,
                "volume_profile": True,
                "market_regime": True
            },
            "scraper": {
                "enabled": True,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
                "proxy": None,
                "timeout": 10,
                "max_retries": 3
            },
            "websocket": {
                "enabled": True,
                "reconnect_delay": 5,
                "heartbeat_interval": 30
            }
        },
        "execution": {
            "enabled": True,
            "paper_trading": True,
            "high_frequency": True,
            "order_types": {
                "entry": "market",
                "exit": "market"
            },
            "order_expiration": {
                "limit": 300,
                "stop": 86400
            },
            "retry_on_failure": True,
            "max_retries": 3,
            "retry_delay": 2,
            "allow_partial_fills": True,
            "min_execution_size": 1,
            "max_slippage_pct": 0.05,
            "execution_algo": "adaptive",
            "order_timing": {
                "allow_overnight": True,
                "execution_hours": {
                    "start": "00:00",
                    "end": "23:59"
                }
            },
            "hf_parameters": {
                "use_microstructure": True,
                "execution_quality_threshold": 0.6,
                "aggressive_threshold": 0.8,
                "passive_threshold": 0.4,
                "impact_model": "square_root",
                "max_slippage_bps": 10,
                "latency_target_ms": 15
            },
            "broker": {
                "name": "paper",
                "api_key": "",
                "api_secret": "",
                "account_id": "",
                "base_url": "",
                "websocket_url": ""
            }
        },
        "risk": {
            "max_position_size": 10,
            "max_positions": 1,
            "max_daily_trades": 20,
            "max_drawdown_pct": 5.0,
            "max_daily_loss_pct": 2.0,
            "max_portfolio_risk_pct": 3.0,
            "risk_per_trade_pct": 0.5,
            "position_sizing": {
                "method": "risk_based",
                "fixed_size": 1,
                "pct_equity": 0.05,
                "risk_factor": 1.0,
                "max_position_pct": 0.5,
                "min_position_size": 1,
                "size_rounding": 1
            },
            "stop_loss": {
                "enabled": True,
                "type": "atr",
                "atr_multiple": 2.0,
                "fixed_pct": 1.0,
                "fixed_points": 50,
                "max_loss_pct": 2.0,
                "trailing": {
                    "enabled": True,
                    "activation_pct": 0.5,
                    "trail_pct": 0.5
                }
            },
            "take_profit": {
                "enabled": True,
                "type": "risk_multiple",
                "risk_reward": 2.0,
                "fixed_pct": 2.0,
                "fixed_points": 100,
                "partial_exits": [
                    {"pct": 0.5, "threshold": 1.0},
                    {"pct": 0.3, "threshold": 1.5}
                ]
            },
            "time_exit": {
                "enabled": True,
                "max_holding_period": 480, # minutes
                "end_of_day": True,
                "eod_time": "15:45"
            },
            "circuit_breakers": {
                "enabled": True,
                "daily_loss_threshold": 3.0,
                "drawdown_threshold": 5.0,
                "consecutive_losses": 5,
                "pause_duration": 60 # minutes
            },
            "hf_risk": {
                "microstructure_limits": {
                    "max_adverse_ticks": 5,           
                    "execution_timeout_seconds": 5,   
                    "spread_factor_limit": 1.5,       
                    "impact_threshold": 0.0005,       
                    "slippage_control": True,        
                    "liquidity_seeking": True        
                },
                "position_decay": {
                    "enabled": True,                  
                    "half_life_seconds": 300,         
                    "min_decay_factor": 0.5,         
                    "use_market_time": True           
                },
                "tick_level_stops": {
                    "enabled": True,                  
                    "atr_multiplier": 0.5,           
                    "tick_movements": 5,              
                    "accumulated_imbalance": 0.7     
                }
            }
        },
        "monitoring": {
            "enabled": True,
            "metrics_update_frequency": 60,
            "health_check_interval": 300,
            "alerts": {
                "enabled": True,
                "methods": ["console", "log", "email"],
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender": "",
                    "password": "",
                    "recipients": []
                },
                "thresholds": {
                    "drawdown": 5.0,
                    "consecutive_losses": 3,
                    "profit_target": 5.0,
                    "high_latency_ms": 100
                }
            },
            "metrics": {
                "performance": True,
                "risk": True,
                "execution": True,
                "model": True,
                "system": True
            },
            "dashboard": {
                "enabled": True,
                "host": "localhost",
                "port": 8050,
                "refresh_interval": 5,
                "charts": ["equity", "drawdown", "positions", "signals", "executions", "metrics"]
            },
            "logging": {
                "trades": True,
                "signals": True,
                "performance": True,
                "execution": True
            }
        },
        "compliance": {
            "enabled": True,
            "log_dir": "logs/compliance",
            "report_dir": "reports/compliance",
            "order_records_dir": "data/orders",
            "audit_trail": True,
            "hf_compliance": {
                "market_manipulation": {
                    "enabled": True,
                    "check_layering": True,
                    "check_spoofing": True,
                    "check_momentum_ignition": True,
                    "check_wash_trades": True,
                    "max_order_cancellation_ratio": 0.95,
                    "min_order_lifespan_ms": 50,
                    "max_price_deviation": 0.003,
                    "quote_stuffing_threshold": 20
                },
                "circuit_breakers": {
                    "enabled": True,
                    "max_orders_per_second": 15,
                    "max_positions_per_minute": 10,
                    "max_notional_per_min": 1000000,
                    "daily_notional_limit": 20000000,
                    "pause_after_consecutive_losses": 5,
                    "pause_duration_minutes": 15
                },
                "record_keeping": {
                    "enabled": True,
                    "order_records": True,
                    "execution_records": True,
                    "strategy_parameters": True,
                    "risk_parameters": True,
                    "market_data_snapshots": True,
                    "retention_days": 730,
                    "hash_verification": True
                },
                "stress_testing": {
                    "enabled": True,
                    "periodic_tests": True,
                    "test_frequency_days": 30,
                    "scenarios": ["high_volatility", "flash_crash", "liquidity_dry_up", "extreme_latency"]
                },
                "reporting": {
                    "enabled": True,
                    "trading_summary": True,
                    "compliance_alerts": True,
                    "risk_metrics": True,
                    "suspicious_activity": True,
                    "recipients": []
                }
            },
            "max_message_rate": 100,
            "max_order_value": 1000000,
            "restricted_instruments": [],
            "trading_hours": {
                "enforce": True,
                "allow_extended_hours": True,
                "regular_start": "09:30",
                "regular_end": "16:00",
                "extended_start": "04:00",
                "extended_end": "20:00"
            },
            "order_throttling": {
                "enabled": True,
                "max_orders_per_second": 15,
                "cooldown_factor": 2,
                "adaptive": True
            }
        }
    }
    
    return default_config

def save_config(config, config_path):
    """Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to configuration file
        
    Returns:
        bool: Success flag
    """
    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save configuration based on file extension
        if config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif config_path.endswith(('.yaml', '.yml')):
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            # Default to JSON
            if not config_path.endswith('.json'):
                config_path += '.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False

def load_components(config):
    """Load system components based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: System components
    """
    try:
        # Initialize components dictionary
        components = {}
        
        # Load data scraper
        try:
            from nq_elite.data.nq_futures_scraper import NQFuturesScraper
            components['scraper'] = NQFuturesScraper(config['data'])
        except ImportError:
            logging.warning("NQFuturesScraper not found, data collection will be unavailable")
        
        # Load feature engineering
        try:
            from nq_elite.data.nq_feature_engineering import NQFeaturesProcessor
            components['features'] = NQFeaturesProcessor(config['data']['features'])
        except ImportError:
            logging.warning("NQFeaturesProcessor not found, feature engineering will be unavailable")
        
        # Load model
        try:
            if config['strategy']['model_params']['model_type'] == 'transformer_hf':
                from nq_elite.models.nq_transformer import NQTransformerModel
                components['model'] = NQTransformerModel(config['strategy']['model_params'])
            elif config['strategy']['model_params']['model_type'] == 'ensemble':
                from nq_elite.models.nq_ensemble import NQEnsembleModel
                components['model'] = NQEnsembleModel(config['strategy']['model_params'])
            else:
                from nq_elite.models.nq_model_factory import create_model
                components['model'] = create_model(config['strategy']['model_params'])
        except ImportError:
            logging.warning("Model modules not found, prediction will be unavailable")
        
        # Load risk manager
        try:
            from nq_elite.risk.adaptive_risk_manager import AdaptiveRiskManager
            components['risk_manager'] = AdaptiveRiskManager(config['risk'])
        except ImportError:
            logging.warning("AdaptiveRiskManager not found, risk management will be unavailable")
        
        # Load trader
        try:
            from nq_elite.execution.nq_live_trader import NQLiveTrader
            components['trader'] = NQLiveTrader(config)
        except ImportError:
            logging.error("NQLiveTrader not found, trading system cannot operate")
            return None
        
        # Load compliance manager
        try:
            from nq_elite.regulatory.compliance_manager import ComplianceManager
            components['compliance'] = ComplianceManager(config['compliance'])
        except ImportError:
            logging.warning("ComplianceManager not found, regulatory compliance will be unavailable")
        
        # Load monitoring
        if config['monitoring']['enabled']:
            try:
                from nq_elite.monitoring.performance_monitor import PerformanceMonitor
                components['monitor'] = PerformanceMonitor(config['monitoring'])
            except ImportError:
                logging.warning("PerformanceMonitor not found, performance monitoring will be unavailable")
            
            # Load dashboard if enabled
            if config['monitoring']['dashboard']['enabled']:
                try:
                    from nq_elite.monitoring.dashboard import Dashboard
                    components['dashboard'] = Dashboard(config['monitoring']['dashboard'])
                except ImportError:
                    logging.warning("Dashboard not found, web interface will be unavailable")
        
        return components
        
    except Exception as e:
        logging.error(f"Error loading components: {str(e)}")
        traceback.print_exc()
        return None

def init_system(components, config):
    """Initialize system components
    
    Args:
        components: System components
        config: Configuration dictionary
        
    Returns:
        bool: Success flag
    """
    try:
        # Set cross-references between components
        
        # Set trader references
        if 'trader' in components:
            if 'scraper' in components:
                components['trader'].set_scraper(components['scraper'])
            
            if 'features' in components:
                components['trader'].set_features_processor(components['features'])
            
            if 'model' in components:
                components['trader'].set_model(components['model'])
            
            if 'risk_manager' in components:
                components['trader'].set_risk_manager(components['risk_manager'])
            
            if 'compliance' in components:
                components['trader'].set_compliance_manager(components['compliance'])
        
        # Set risk manager references
        if 'risk_manager' in components:
            if 'trader' in components:
                components['risk_manager'].set_trader(components['trader'])
        
        # Set monitor references
        if 'monitor' in components:
            if 'trader' in components:
                components['monitor'].set_trader(components['trader'])
            
            if 'risk_manager' in components:
                components['monitor'].set_risk_manager(components['risk_manager'])
        
        # Set dashboard references
        if 'dashboard' in components:
            if 'trader' in components:
                components['dashboard'].set_trader(components['trader'])
            
            if 'monitor' in components:
                components['dashboard'].set_monitor(components['monitor'])
        
        # Initialize data
        if 'scraper' in components:
            if not components['scraper'].start():
                logging.error("Failed to start data scraper")
                return False
            
            # Wait for initial data
            logging.info("Waiting for initial data...")
            time.sleep(5)
        
        # Load model
        if 'model' in components:
            model_path = config['strategy']['model_params'].get('model_path')
            if model_path and os.path.exists(model_path):
                if not components['model'].load(model_path):
                    logging.error(f"Failed to load model from {model_path}")
                    return False
            else:
                logging.warning(f"Model path not found: {model_path}")
        
        # Start monitoring
        if 'monitor' in components:
            components['monitor'].start()
        
        # Start dashboard
        if 'dashboard' in components and config['monitoring']['dashboard']['enabled']:
            dashboard_thread = threading.Thread(
                target=components['dashboard'].start,
                args=(),
                daemon=True
            )
            dashboard_thread.start()
            
            # Store thread in components
            components['dashboard_thread'] = dashboard_thread
            
            logging.info(f"Dashboard started at http://{config['monitoring']['dashboard']['host']}:{config['monitoring']['dashboard']['port']}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error initializing system: {str(e)}")
        traceback.print_exc()
        return False

def start_trading(components, config):
    """Start trading system
    
    Args:
        components: System components
        config: Configuration dictionary
        
    Returns:
        bool: Success flag
    """
    try:
        if 'trader' not in components:
            logging.error("Trader component not found")
            return False
        
        # Start trader
        if not components['trader'].start():
            logging.error("Failed to start trader")
            return False
        
        logging.info("Trading system started successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error starting trading: {str(e)}")
        traceback.print_exc()
        return False

def stop_system(components):
    """Stop system components
    
    Args:
        components: System components
        
    Returns:
        bool: Success flag
    """
    try:
        logging.info("Stopping NQ Alpha Elite trading system...")
        
        # Stop trader
        if 'trader' in components:
            components['trader'].stop()
        
        # Stop scraper
        if 'scraper' in components:
            components['scraper'].stop()
        
        # Stop monitor
        if 'monitor' in components:
            components['monitor'].stop()
        
        # Stop dashboard
        if 'dashboard' in components:
            components['dashboard'].stop()
        
        # Stop compliance manager
        if 'compliance' in components:
            components['compliance'].stop()
        
        logging.info("Trading system stopped successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error stopping system: {str(e)}")
        traceback.print_exc()
        return False

def signal_handler(signum, frame):
    """Handle termination signals
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logging.info(f"Received signal {signum}, shutting down...")
    
    # Stop system
    if 'system_components' in globals():
        stop_system(system_components)
    
    # Exit with success status
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def deploy_trader(config_path=None):
    """Deploy NQ Alpha Elite trading system
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        int: Exit code
    """
    try:
        # Setup logging
        logger = setup_logging()
        
        # Print welcome message
        print("\n" + "=" * 80)
        print("  NQ Alpha Elite v7.0.0 - Advanced Algorithmic Trading System")
        print("  High-Frequency Trading for Nasdaq 100 E-mini Futures")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80 + "\n")
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Load configuration
        if config_path:
            config = load_config(config_path)
            if config is None:
                return 1
        else:
            # Create default configuration
            config = create_default_config()
            
            # Save default configuration
            default_config_path = "configs/nq_elite_default_config.json"
            if not os.path.exists(default_config_path):
                save_config(config, default_config_path)
                logger.info(f"Default configuration saved to {default_config_path}")
        
        # Create data and logs directories
        os.makedirs(config['system']['data_dir'], exist_ok=True)
        os.makedirs(config['system']['log_dir'], exist_ok=True)
        os.makedirs(config['system']['models_dir'], exist_ok=True)
        
        # Load components
        logger.info("Loading system components...")
        components = load_components(config)
        if components is None:
            logger.error("Failed to load system components")
            return 1
        
        # Make components available to signal handler
        global system_components
        system_components = components
        
        # Initialize system
        logger.info("Initializing system...")
        if not init_system(components, config):
            logger.error("Failed to initialize system")
            stop_system(components)
            return 1
        
        # Start trading
        logger.info("Starting trading system...")
        if not start_trading(components, config):
            logger.error("Failed to start trading")
            stop_system(components)
            return 1
        
        # Keep system running until terminated
        logger.info("NQ Alpha Elite system is running")
        logger.info("Press Ctrl+C to stop")
        
        # Wait for dashboard thread to exit if dashboard is running
        if 'dashboard_thread' in components:
            components['dashboard_thread'].join()
        else:
            # Otherwise, just keep the main thread running
            while True:
                time.sleep(1)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error deploying trader: {str(e)}")
        traceback.print_exc()
        
        # Attempt to stop system
        if 'system_components' in globals():
            stop_system(system_components)
        
        return 1

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NQ Alpha Elite v7.0.0 - Deploy Trading System")
    parser.add_argument('--config', '-c', type=str, help="Path to configuration file")
    parser.add_argument('--generate-config', '-g', action='store_true', help="Generate default configuration")
    parser.add_argument('--output', '-o', type=str, help="Output path for generated configuration")
    
    args = parser.parse_args()
    
    # Generate default configuration if requested
    if args.generate_config:
        config = create_default_config()
        output_path = args.output or "configs/nq_elite_config.json"
        if save_config(config, output_path):
            print(f"Default configuration generated at {output_path}")
            return 0
        else:
            return 1
    
    # Deploy trader
    return deploy_trader(args.config)

if __name__ == "__main__":
    sys.exit(main())
