# Import-free AST contract, native source sha256 108ceff11e5a5c9a34a57bbc380447cf9941beb673a1bd421abc818de54ccd5c
class MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3ImageToVideo', display_name='MiniMax H3 Image to Video', category='model/conditioning/minimax', inputs=[io.Clip.Input('clip'), io.Vae.Input('vae'), io.String.Input('prompt', multiline=True, dynamic_prompts=True), io.Int.Input('width', default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32), io.Int.Input('height', default=768, min=32, max=nodes.MAX_RESOLUTION, step=32), io.Int.Input('length', default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"), io.Image.Input('first_frame', optional=True), io.Image.Input('last_frame', optional=True)], outputs=[io.Conditioning.Output(display_name='positive'), io.Latent.Output()])

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length, first_frame=None, last_frame=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)
        images = []
        keyframes = []
        if first_frame is not None:
            img = _resize(first_frame[:1], width, height, 'disabled')
            images.append(img)
            keyframes.append({'resolved_frame_index': 0, 'image': img})
        if last_frame is not None:
            img = _resize(last_frame[:1], width, height, 'center')
            images.append(img)
            keyframes.append({'resolved_frame_index': frame_count - 1, 'image': img})
        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for kf in keyframes:
                kf['latent'] = vae.encode(kf.pop('image'))
            cond = node_helpers.conditioning_set_values(cond, {'minimax_keyframes': keyframes})
        return io.NodeOutput(cond, latent)
