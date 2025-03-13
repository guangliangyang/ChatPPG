
#TimeLLM add a linera layer to insurance important information cannot be loss
if self.down_proj:
    dec_out = self.down_proj(dec_out)
else:
    dec_out = dec_out[:, :, :self.d_ff]
