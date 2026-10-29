"""
Proactive Maintenance Scheduling & Health Index Estimator
Author: Batch 11 - Predictive Maintenance System
Aligned with:
- SDG 9: Industry, Innovation and Infrastructure (Uptime & Reliability)
- SDG 12: Responsible Consumption and Production (Preventing Waste & Resource Overconsumption)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class MaintenanceScheduler:
    def __init__(self, threshold_warning: float, threshold_critical: float):
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical

    def compute_health_index(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """
        Calculates the Equipment Health Index (EHI) [0% to 100%].
        100% = Peak pristine condition
        80-100% = Healthy operational regime
        60-80% = Minor wear & tear
        40-60% = Warning state / incipient failure
        <40% = Critical failure state
        """
        # Linear degradation mapping from error to 0-100 scale
        ratio = reconstruction_errors / max(self.threshold_critical, 1e-6)
        health_index = np.clip(100.0 * (1.0 - 0.6 * ratio), 0.0, 100.0)
        return health_index

    def estimate_remaining_useful_life(
        self, health_index_series: np.ndarray, time_step_hours: float = 1.0
    ) -> Optional[float]:
        """
        Linear regression trend over the recent degrading window to estimate
        Remaining Useful Life (RUL in hours) before reaching critical failure threshold (40%).
        """
        if len(health_index_series) < 5:
            return None

        recent = health_index_series[-30:]
        x = np.arange(len(recent)) * time_step_hours
        y = recent

        if np.std(x) > 0 and np.std(y) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            if slope < -1e-4:
                critical_threshold = 40.0
                time_to_critical = (critical_threshold - intercept) / slope
                rul_hours = max(0.5, float(time_to_critical - x[-1]))
                return round(rul_hours, 1)

        return None

    def generate_maintenance_ticket(
        self,
        current_error: float,
        health_index: float,
        rul_hours: Optional[float],
        current_timestamp: str,
    ) -> Dict[str, Any]:
        """
        Generates a demonstration maintenance work order for the current model assessment.
        """
        if health_index < 40.0 or current_error >= self.threshold_critical:
            priority = "CRITICAL - IMMEDIATE ACTION"
            action = "Dispatch maintenance crew immediately. Inspect compressor valves, seals, and pneumatic manifold."
            lead_time = f"{rul_hours:.1f} hours" if rul_hours is not None else "Immediate inspection"
            sdg_impact = "Prevents catastrophic compressor seizure and sudden train service shutdown (SDG 9)."
        elif health_index < 70.0 or current_error >= self.threshold_warning:
            priority = "WARNING - SCHEDULED REPAIR"
            action = "Schedule targeted service within next 24-48 hours during off-peak window. Inspect air dryer towers."
            lead_time = f"{rul_hours:.1f} hours" if rul_hours is not None else "Not estimable from current trend"
            sdg_impact = "Avoids secondary damage to downstream pneumatic braking systems (SDG 12)."
        elif health_index < 85.0:
            priority = "ADVISORY - PLANNED INSPECTION"
            action = "Add to routine weekly maintenance checklist. Verify lubrication oil levels and motor current draw."
            lead_time = f"{rul_hours:.1f} hours" if rul_hours is not None else "Not estimable from current trend"
            sdg_impact = "Optimizes component lifespan and prevents premature parts replacement (SDG 12)."
        else:
            priority = "NORMAL - OPTIMAL HEALTH"
            action = "Normal operation. No maintenance required."
            lead_time = "Not estimable while operation remains stable"
            sdg_impact = "Maximizes energy efficiency and equipment uptime (SDG 9)."

        return {
            "Timestamp": current_timestamp,
            "Health_Index": f"{health_index:.1f}%",
            "Reconstruction_Error": f"{current_error:.5f}",
            "Estimated_RUL": lead_time,
            "Priority": priority,
            "Recommended_Action": action,
            "SDG_Contribution": sdg_impact
        }
