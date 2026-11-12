"""YOLO object detection on request, answering with the detected objects and
saying them."""

# %% imports
import os
import sys
from dataclasses import dataclass

import numpy
from sensor_msgs.msg import CameraInfo

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.detected_objects_announcer import DetectedObjectsAnnouncer
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


# %% analysis engine
@dataclass
class AnalysisEngine(AnalysisEngineInterface):
    """Detect objects in the current frame of the HSR head camera for each
    query, reply with them and publish a sentence naming them for the
    text-to-speech."""

    def name(self) -> str:
        return "live_yolo"

    def implementation(self) -> Pipeline:
        kinect_info = CameraInfo(
            width=640,
            height=480,
            k=[525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0],
            p=[525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        camera = CollectionReaderDescriptorFactory.create_descriptor(
            "opencv",
            device=2,
            camera_info=kinect_info,
            depth=numpy.full((480, 640), 1000, dtype=numpy.uint16),
        )
        yolo = YoloAnnotator.Descriptor()
        yolo.parameters.pretrained_model = "ultimate_model_v8_1.26.2.pt"
        yolo.parameters.threshold = 0.6
        yolo.parameters.ignored_class_names = ["person", "dining table"]

        pipeline = Pipeline("LiveYOLO")
        pipeline.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=camera),
                ImagePreprocessorAnnotator(),
                YoloAnnotator(descriptor=yolo),
                DetectedObjectsAnnouncer(),
                GenerateQueryResult(),
                ActionServerCheck(),
            ]
        )
        return pipeline


# %% direct start
def main() -> None:
    """Start RoboKudo with the YOLO object detection analysis engine."""
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "robokudo.scripts.main",
            "_ae",
            "yolo_object_detection",
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
