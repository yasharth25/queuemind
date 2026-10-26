from src.vision.privacy import load_privacy_config
def test_privacy_defaults_disable_video_retention():
 config=load_privacy_config(); assert config["save_processed_video"] is False and config["anonymous_tracking_only"] is True
