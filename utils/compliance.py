"""
utils/compliance.py
===================
PPE Compliance Checker.

For each detected 'person', finds which PPE items are spatially
associated (worn), then checks against zone compliance rules.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import yaml


class ComplianceChecker:
    """
    Determines whether each person in the frame is wearing required PPE.

    Uses spatial IoU/overlap to associate PPE bounding boxes with person boxes.
    Then checks associated PPE against zone-specific rules.
    """

    PERSON_CLASS = "person"
    PPE_CLASSES = {
        "hard-hat", "safety-vest", "gloves",
        "safety-goggles", "face-mask", "safety-boots",
        "harness", "ear-protection"
    }

    # Body region vertical fractions (top of person bbox = 0, bottom = 1)
    BODY_REGIONS = {
        "head":   (0.0, 0.30),   # top 30%
        "torso":  (0.20, 0.75),  # middle
        "hands":  (0.40, 0.85),  # lower sides
        "feet":   (0.75, 1.00),  # bottom 25%
    }

    PPE_BODY_MAP = {
        "hard-hat":       "head",
        "safety-goggles": "head",
        "ear-protection": "head",
        "face-mask":      "head",
        "safety-vest":    "torso",
        "harness":        "torso",
        "gloves":         "hands",
        "safety-boots":   "feet",
    }

    def __init__(self, config_path: str = "config/ppe_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.assoc_distance = self.cfg.get("association_distance", 0.7)
        self.compliance_rules = self.cfg.get("compliance_rules", {})
        self.active_zone = "default"

    def set_zone(self, zone: str):
        """Switch active compliance zone."""
        if zone in self.compliance_rules:
            self.active_zone = zone
            print(f"🏭 Zone changed to: {zone}")
        else:
            print(f"⚠️  Zone '{zone}' not found. Using 'default'.")
            self.active_zone = "default"

    def check_frame(self, detections: List[Dict]) -> List[Dict]:
        """
        Check PPE compliance for all persons in the frame.

        Args:
            detections: List of detection dicts from PPEDetector

        Returns:
            List of person compliance results:
            [
                {
                    'track_id': int,
                    'person_bbox': [x1,y1,x2,y2],
                    'worn_ppe': set of class names,
                    'missing_ppe': set of class names,
                    'compliant': bool,
                    'compliance_score': float (0-100),
                    'status_text': str,
                }
            ]
        """
        persons = [d for d in detections if d["class_name"] == self.PERSON_CLASS]
        ppe_items = [d for d in detections if d["class_name"] in self.PPE_CLASSES]

        results = []
        for person in persons:
            worn = self._associate_ppe(person, ppe_items)
            compliance = self._evaluate_compliance(worn)
            results.append({
                "track_id": person.get("track_id", -1),
                "person_bbox": person["bbox"],
                "person_center": person["center"],
                "worn_ppe": worn,
                "missing_ppe": compliance["missing"],
                "recommended_missing": compliance["recommended_missing"],
                "compliant": compliance["compliant"],
                "compliance_score": compliance["score"],
                "status_text": compliance["status_text"],
                "zone": self.active_zone,
            })

        return results

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _associate_ppe(self, person: Dict, ppe_items: List[Dict]) -> Set[str]:
        """
        Associate PPE items with a person using spatial overlap + body region.

        Strategy:
        1. Check if PPE center is INSIDE person bbox (expanded slightly)
        2. Additionally check IoU > threshold
        3. Check if PPE is in correct body region
        """
        worn = set()
        px1, py1, px2, py2 = person["bbox"]
        pw = px2 - px1
        ph = py2 - py1

        # Expand person bbox slightly for association
        margin_x = pw * 0.15
        margin_y = ph * 0.10
        epx1 = px1 - margin_x
        epx2 = px2 + margin_x
        epy1 = py1 - margin_y
        epy2 = py2 + margin_y

        for ppe in ppe_items:
            qx1, qy1, qx2, qy2 = ppe["bbox"]
            qcx, qcy = ppe["center"]

            # Check 1: PPE center inside expanded person box
            center_inside = (epx1 <= qcx <= epx2) and (epy1 <= qcy <= epy2)

            # Check 2: IoU overlap
            iou = self._compute_iou(
                [px1, py1, px2, py2],
                [qx1, qy1, qx2, qy2]
            )

            # Check 3: Body region match
            region_ok = self._check_body_region(
                ppe["class_name"], qcy, py1, py2
            )

            # Accept if center is inside AND (iou > 0.05 OR region matches)
            if center_inside and (iou > 0.02 or region_ok):
                worn.add(ppe["class_name"])

        return worn

    def _evaluate_compliance(self, worn_ppe: Set[str]) -> Dict:
        """Check worn PPE against zone rules."""
        zone_rules = self.compliance_rules.get(
            self.active_zone,
            self.compliance_rules.get("default", {})
        )

        required = set(zone_rules.get("required", []))
        recommended = set(zone_rules.get("recommended", []))

        missing_required = required - worn_ppe
        missing_recommended = recommended - worn_ppe

        compliant = len(missing_required) == 0

        # Score: 100 if all required present
        if len(required) > 0:
            score = ((len(required) - len(missing_required)) / len(required)) * 100
        else:
            score = 100.0

        if compliant:
            status = "✅ COMPLIANT"
        else:
            missing_str = ", ".join(sorted(missing_required))
            status = f"❌ MISSING: {missing_str}"

        return {
            "compliant": compliant,
            "missing": missing_required,
            "recommended_missing": missing_recommended,
            "score": round(score, 1),
            "status_text": status,
        }

    def _check_body_region(
        self, ppe_name: str, ppe_cy: float, person_top: float, person_bottom: float
    ) -> bool:
        """Check if PPE center Y is in the expected body region."""
        region_name = self.PPE_BODY_MAP.get(ppe_name)
        if region_name is None:
            return True  # No region constraint

        region = self.BODY_REGIONS.get(region_name, (0, 1))
        person_height = person_bottom - person_top
        region_top = person_top + region[0] * person_height
        region_bot = person_top + region[1] * person_height

        return region_top <= ppe_cy <= region_bot

    @staticmethod
    def _compute_iou(box1: List[int], box2: List[int]) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
