"""Native SAM 3D Body video-to-mesh workflow."""
from core import WorkflowGraph


BODY_MODEL = "sam_3d_body_dinov3_bf16.safetensors"
TRACK_MODEL = "sam3.1_multiplex_fp16.safetensors"
MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"
DETECTOR_MODEL = "rt_detr_v4-x-hgnet_fp32.safetensors"


def sam3d_video_to_body(video_filename, seconds=5.0, start_time=0.0,
                       prompt="person", filename_prefix="video/SAM3D_body",
                       batch_size=4, moge_batch_size=1, tracking=True,
                       moge=True, face_expression=True, hand_refinement=True,
                       overlay=True, keep_audio=True, export_style="body_mesh",
                       detection_threshold=0.5, max_people=4,
                       body_model=BODY_MODEL, track_model=TRACK_MODEL,
                       moge_model=MOGE_MODEL, detector_model=DETECTOR_MODEL):
    """Extract an animated GLB and render an overlay MP4 at the source FPS."""
    if not video_filename:
        raise ValueError("sam3d requires --video (local file or ComfyUI input name)")
    if seconds < 0 or start_time < 0:
        raise ValueError("--seconds and --start-time must be nonnegative")
    if not 1 <= batch_size <= 512 or not 1 <= moge_batch_size <= 64:
        raise ValueError("--batch-size must be 1..512; --moge-batch-size must be 1..64")
    if not 0 <= detection_threshold <= 1 or not 1 <= max_people <= 64:
        raise ValueError("--detection-threshold must be 0..1; --max-people must be 1..64")
    if export_style not in {"body_mesh", "scail"}:
        raise ValueError("--export-style must be body_mesh or scail")

    graph = WorkflowGraph()
    video = graph.node("LoadVideo", file=video_filename)
    if seconds or start_time:
        video = graph.node("Video Slice", video=video, start_time=start_time,
                           duration=seconds, strict_duration=False)
    components = graph.node("GetVideoComponents", video=video)
    images = components[0]
    body = graph.node("SAM3DBody_Loader", model_file=body_model)
    detector = graph.node("UNETLoader", unet_name=detector_model,
                          weight_dtype="default")
    boxes = graph.node("RTDETR_detect", model=detector, image=images,
                       threshold=detection_threshold, class_name="person",
                       max_detections=max_people)
    predict_inputs = dict(sam3d_body_model=body, image=images,
                          run_hand_refinement=hand_refinement, fov=0.0,
                          batch_size=batch_size, bboxes=boxes)
    if tracking:
        checkpoint = graph.node("CheckpointLoaderSimple", ckpt_name=track_model)
        conditioning = graph.node("CLIPTextEncode", clip=checkpoint[1], text=prompt)
        tracks = graph.node("SAM3_VideoTrack", images=images, model=checkpoint[0],
                            conditioning=conditioning,
                            detection_threshold=detection_threshold,
                            max_objects=max_people, detect_interval=1)
        predict_inputs["track_data"] = tracks
    if moge:
        geometry_model = graph.node("LoadMoGeModel", model_name=moge_model)
        geometry = graph.node("MoGeInference", moge_model=geometry_model,
                              image=images, resolution_level=9, fov_x_degrees=0.0,
                              batch_size=moge_batch_size, force_projection=True,
                              apply_mask=True, refine_steps=3)
        fov = graph.node("MoGeGeometryToFOV", moge_geometry=geometry,
                         axis="vertical", unit="degrees")
        predict_inputs["fov"] = fov
    pose = graph.node("SAM3DBody_Predict", **predict_inputs)
    if face_expression:
        pose = graph.node("SAM3DBody_FaceExpression", sam3d_body_model=body,
                          mhr_pose_data=pose, image=images, strength=1.0,
                          mouth_strength=1.0, eye_strength=2.0, brow_strength=2.0,
                          input_threshold=0.02, blendshape_smooth_window=7)
    pose = graph.node("SAM3DBody_Smooth", mhr_pose_data=pose, strength=1.0,
                      method="gaussian", window=7, rotation_threshold_degrees=15.0)
    export_inputs = {
        "pose_data": pose, "sam3d_body_model": body, "fps": components[2],
        "camera_translation": "off", "track_index": -1,
        "format": "glb", "format.mesh_style": export_style,
        "format.bone_smooth_window": 0,
    }
    if export_style == "body_mesh":
        export_inputs.update({"format.mesh_style.bone_vis": "off",
                              "format.mesh_style.shader": "default"})
    else:
        export_inputs.update({
            "format.mesh_style.stick_radius_m": 0.022,
            "format.mesh_style.marker_radius_m": 0.0,
            "format.mesh_style.material_roughness": 0.3,
            "format.mesh_style.include_hands": False,
            "format.mesh_style.hand_marker_radius_m": 0.005,
            "format.mesh_style.hand_stick_radius_m": 0.003,
            "format.mesh_style.face_style": "full",
        })
    mesh = graph.node("BuildPoseFile", **export_inputs)
    graph.node("Preview3D", model_file=mesh)
    render_inputs = {
        "pose_data": pose, "width": 0, "height": 0, "render_style": "mesh",
        "render_style.shader": "default", "render_style.opacity": 1.0,
        "render_style.person_palette_falloff": 0.6,
        "render_style.region": "full_body",
    }
    if overlay:
        render_inputs["background"] = images
    rendered = graph.node("SAM3DBody_Render", **render_inputs)
    create_inputs = dict(images=rendered, fps=components[2],
                         bit_depth="auto", color_space=components[4])
    if keep_audio:
        create_inputs["audio"] = components[1]
    output = graph.node("CreateVideo", **create_inputs)
    graph.node("SaveVideo", video=output, filename_prefix=filename_prefix,
               format="auto", codec="h264")
    return graph.to_dict()
