for f in ['/opt/work/SpatialLM/spatiallm/model/spatiallm_qwen.py', '/opt/work/SpatialLM/spatiallm/model/spatiallm_llama.py']:
    s = open(f).read()
    if 'enable_flash=False' not in s:
        s = s.replace('                enable_fourier_encode=True,\n', '                enable_fourier_encode=True,\n                enable_flash=False,\n')
    open(f, 'w').write(s); print(f, s.count('enable_flash=False'))
