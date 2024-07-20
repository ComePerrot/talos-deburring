from pathlib import Path

absolute_path = Path(__file__).resolve().parent
urdf_path = {
    "example_robot_data":str(absolute_path / "urdf/talos_example_robot.urdf"),
    "public_pal":str(absolute_path / "urdf/talos_public.urdf"),
}

srdf_path = str(absolute_path / "srdf/talos.srdf")
