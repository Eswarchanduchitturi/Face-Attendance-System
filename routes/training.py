import os

from flask import Blueprint, jsonify, render_template, request


def create_training_blueprint(training_jobs):
    bp = Blueprint("training", __name__)

    @bp.route("/train")
    def train():
        job_id = training_jobs.start_job(include_smoke_test=False)
        return render_template("train.html", job_id=job_id)

    @bp.route("/train_and_test")
    def train_and_test():
        try:
            if not os.path.exists("TrainingImage"):
                return jsonify({"status": "fail", "error": "No training images found"})
            images = [f for f in os.listdir("TrainingImage") if f.endswith(".jpg")]
            if len(images) == 0:
                return jsonify({"status": "fail", "error": "No face images available"})

            job_id = training_jobs.start_job(include_smoke_test=True)
            return jsonify({"status": "started", "job_id": job_id})
        except Exception as e:
            return jsonify({"status": "fail", "error": str(e)})

    @bp.route("/train_test_progress")
    def train_test_progress():
        try:
            job_id = request.args.get("job_id")
            job = training_jobs.get_job(job_id) if job_id else training_jobs.get_active_job()
            if not job:
                return jsonify({
                    "status": "idle",
                    "progress": 0,
                    "stage": "idle",
                    "message": "No active training job"
                })

            if job["status"] == "fail":
                return jsonify({
                    "status": "fail",
                    "job_id": job["id"],
                    "progress": job["progress"],
                    "stage": job["stage"],
                    "message": job["message"],
                    "error": job.get("error")
                })

            if job["status"] == "success":
                return jsonify({
                    "status": "success",
                    "job_id": job["id"],
                    "progress": job["progress"],
                    "stage": job["stage"],
                    "message": job["message"]
                })

            return jsonify({
                "status": "running",
                "job_id": job["id"],
                "progress": job["progress"],
                "stage": job["stage"],
                "message": job["message"]
            })
        except Exception as e:
            return jsonify({"status": "fail", "error": str(e)})

    return bp
