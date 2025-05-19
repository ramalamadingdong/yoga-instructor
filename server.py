from flask import Flask, request, jsonify

app = Flask(__name__)

# List of yoga positions
yoga_positions = [
    "Mountain Pose (Tadasana)",
    "Downward-Facing Dog (Adho Mukha Svanasana)",
    "Warrior I (Virabhadrasana I)",
    "Warrior II (Virabhadrasana II)",
    "Tree Pose (Vrksasana)",
    "Child's Pose (Balasana)",
    "Cobra Pose (Bhujangasana)",
    "Bridge Pose (Setu Bandha Sarvangasana)",
    "Triangle Pose (Trikonasana)",
    "Corpse Pose (Savasana)"
]

@app.route('/get_position', methods=['GET'])
def get_position():
    import random
    position = random.choice(yoga_positions)
    return jsonify({"position": position}), 200

@app.route('/get_position', methods=['GET'])
def get_pose_hold_instrutions():
    sample_instructions = "as you're holding this pose, breathe in and out slowly and deeply"
    return jsonify({sample_instructions}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run on all interfaces