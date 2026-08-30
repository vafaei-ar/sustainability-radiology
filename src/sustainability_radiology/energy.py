"""Energy normalization utilities for radiology AI benchmarks.

The historical MONAI benchmark scripts processed a fixed number of forward-pass
iterations. Each iteration processed a batch of samples. Population modeling must
normalize total run energy by the total number of processed samples, not by batch
size alone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkRun:
    """One measured inference benchmark run.

    Parameters
    ----------
    total_energy_kwh:
        Total measured operational energy for the run.
    batch_size:
        Number of samples processed per forward pass.
    n_iterations:
        Number of forward-pass iterations executed during the measured run.
    """

    total_energy_kwh: float
    batch_size: int
    n_iterations: int = 5000

    @property
    def n_samples(self) -> int:
        """Total individual samples processed during the run."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        return self.batch_size * self.n_iterations

    @property
    def energy_per_inference_kwh(self) -> float:
        """Measured energy per individual sample/inference."""
        if self.total_energy_kwh < 0:
            raise ValueError("total_energy_kwh cannot be negative")
        return self.total_energy_kwh / self.n_samples

    def energy_per_n_inferences_kwh(self, n: int = 1000) -> float:
        """Energy required for ``n`` individual inferences."""
        if n <= 0:
            raise ValueError("n must be positive")
        return self.energy_per_inference_kwh * n


def carbon_from_energy(
    energy_kwh: float,
    carbon_intensity_kg_per_kwh: float,
) -> float:
    """Convert electricity use to kg CO2-equivalent."""
    if energy_kwh < 0:
        raise ValueError("energy_kwh cannot be negative")
    if carbon_intensity_kg_per_kwh < 0:
        raise ValueError("carbon intensity cannot be negative")
    return energy_kwh * carbon_intensity_kg_per_kwh


def annual_ai_energy_kwh(
    n_exams: float,
    ai_tasks_per_exam: float,
    energy_per_task_kwh: float,
    adoption_fraction: float = 1.0,
    workload_multiplier: float = 1.0,
) -> float:
    """Estimate annual operational AI energy for an imaging workflow.

    ``workload_multiplier`` converts one benchmark input into the computational
    work required for one clinical examination. It must be derived separately
    for each modality/task, for example from number of images, 3D patches, or
    volumes per examination.
    """
    if n_exams < 0 or ai_tasks_per_exam < 0 or energy_per_task_kwh < 0:
        raise ValueError("counts and energy must be non-negative")
    if not 0 <= adoption_fraction <= 1:
        raise ValueError("adoption_fraction must be between 0 and 1")
    if workload_multiplier < 0:
        raise ValueError("workload_multiplier must be non-negative")

    return (
        n_exams
        * adoption_fraction
        * ai_tasks_per_exam
        * workload_multiplier
        * energy_per_task_kwh
    )
