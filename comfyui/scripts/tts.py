"""TTS workflows for ComfyUI (server URL via COMFY_URL, default http://localhost:8188)."""
from core import WorkflowGraph


def qwen_tts(text, voice_instruct="A clear, friendly voice.", language="Auto",
              filename_prefix="tts"):
    g = WorkflowGraph()
    audio = g.node("AILab_Qwen3TTSVoiceDesign", text=text, instruct=voice_instruct,
                   model_size="1.7B", language=language, unload_models=True, seed=-1)
    g.node("SaveAudioMP3", audio=audio[0], filename_prefix=filename_prefix,
           quality="V0", audioUI="")
    return g.to_dict()


def qwen_voice_clone(text, voice_name="gio", language="Auto", filename_prefix="clone"):
    """Clone a named voice and speak text.
    voice_name: a reference audio filename uploaded to ComfyUI inputs, or a name in VOICE_LIBRARY."""
    g = WorkflowGraph()
    ref_audio = g.node("VHS_LoadAudioUpload", audio=voice_name)
    voice = g.node("AILab_Qwen3TTSVoicesLibrary",
                   reference_audio=ref_audio[0], reference_text="",
                   model_size="1.7B", device="auto", precision="bf16",
                   x_vector_only=True, voice_name=voice_name, unload_models=True)
    audio = g.node("AILab_Qwen3TTSVoiceClone", target_text=text, model_size="1.7B",
                   language=language, voice=voice[0], unload_models=True, seed=-1)
    g.node("SaveAudioMP3", audio=audio[0], filename_prefix=filename_prefix,
           quality="V0", audioUI="")
    return g.to_dict()
