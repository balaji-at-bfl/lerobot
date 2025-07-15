Task: Add Simulation Support to lerobot with Shared Control and Visualization Logic

I’m working with the lerobot framework, using SO101 arms — one leader and one follower. I currently have working functionality for:

Teleoperation

Recording datasets


The real-robot setup uses record.py, and for visualization, it uses Rerun (for example, to show camera feeds or state updates).
I also have a MuJoCo simulation model file called so101_new_calib.xml in the simulation/ directory of the repo.

🔍 Before You Begin
Please analyze the current codebase first — including the main scripts (record.py, teleoperate.py, test_policy.py) and any simulation, robot interface, and visualization modules.

Then, plan your approach in detail before modifying anything. Your changes should be modular, flag-driven, and designed to reuse as much of the existing logic (e.g., Rerun streaming, dataset format, policy integration) as possible.

✅ New Functionality to Add
1. Flag-driven simulation mode
Introduce two key flags:

bash
Copy code
--enable-sim=true
--disable-follower=true #  works only with --enable-sim
2. Mode: enable-sim=true (with real leader and real follower)
Use the real leader and follower as usual.

Also spawn a MuJoCo simulation of the follower that mirrors the same actions being sent to the real follower.

The simulated robot should move identically to the real one.

3. Mode: enable-sim=true and disable-follower=true
Connect only to the real leader, and simulate the follower.

Do not attempt to connect to the real follower hardware.

All tools should behave as if the simulated follower is real:

teleoperate.py: Control the simulated robot using the real leader.

record.py:

Collect actions from the real leader.

Collect observations from the MuJoCo simulation.

Save the dataset just as it would for the real robot.

If a policy is passed, run it on the simulated follower and record its behavior.

Important: In this mode, record.py should continue to use the existing Rerun-based visualization, but now showing:

Simulated MuJoCo camera feed in place of the real robot’s camera stream.

Any additional state/pose data from the simulation, mimicking the structure used with the real robot.

📌 Implementation Notes
Ensure the MuJoCo-based robot wrapper exposes camera data and joint states in a form compatible with the rest of the pipeline.

Structure the code so that real and simulated robot interfaces can be used interchangeably based on flags.

Don’t introduce special-case logic where unnecessary — keep the interface unified and clean.

Reuse Rerun, data formats, and action/observation pipelines to avoid duplicating logic between real and sim modes.

Add brief documentation/comments where appropriate.
