#!/usr/bin/env python3
"""Build the pinned LiteRT-LM 0.17.0 Android runner; no Gradle or APK required."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, urllib.request, zipfile

GOOGLE='https://dl.google.com/dl/android/maven2/'
MAVEN='https://repo.maven.apache.org/maven2/'
ARTIFACTS={
 'litertlm.aar': (GOOGLE+'com/google/ai/edge/litertlm/litertlm-android/0.17.0/litertlm-android-0.17.0.aar', '28aa6bc43efcee35b31795f9e5ca633c3dec06d9f3fb85ecb6a753fa360e2134'),
 'kotlin-stdlib.jar': (MAVEN+'org/jetbrains/kotlin/kotlin-stdlib/2.4.0/kotlin-stdlib-2.4.0.jar', 'ccb14ff83fabcb11458b798dbc9824748ccdffeec79c9aba789e6ed1cda86b1c'),
 'kotlin-reflect.jar': (MAVEN+'org/jetbrains/kotlin/kotlin-reflect/2.4.0/kotlin-reflect-2.4.0.jar', 'f0d8e8908624eb4478bb131e9f887366680d2e239741195e548fa5c906963b90'),
 'gson.jar': (MAVEN+'com/google/code/gson/gson/2.14.0/gson-2.14.0.jar', '2cbd119bf1961c28788310963dc80ba65f58cdeec1dd139c8bdb1240faa2c36f'),
 'coroutines-core.jar': (MAVEN+'org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.11.0/kotlinx-coroutines-core-jvm-1.11.0.jar', 'd1d75aa01dffbb4d1c520e67e4c4e7f5f6174718e7cb4632412503f2f0e604fa'),
 'coroutines-android.jar': (MAVEN+'org/jetbrains/kotlinx/kotlinx-coroutines-android/1.11.0/kotlinx-coroutines-android-1.11.0.jar', 'c2cb206d27017c7d1bf5ff179787397543d13748dbabb0d7237e1585e0b29044'),
 'r8.jar': (GOOGLE+'com/android/tools/r8/9.5.16-dev/r8-9.5.16-dev.jar', '2790aca5b86383507cb77906aef4d095a3853dbdd31711dae94f3394e4f2c532'),
}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--android-sdk',type=pathlib.Path,required=True)
    p.add_argument('--jdk',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--android-api',default='36')
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    download=out/'downloads';download.mkdir(exist_ok=True)
    for name,(url,sha) in ARTIFACTS.items():
        path=download/name
        if not path.exists():
            with urllib.request.urlopen(url,timeout=90) as src,path.open('wb') as dst: shutil.copyfileobj(src,dst)
        if digest(path)!=sha: raise ValueError('Checksum mismatch: '+name)
    android=a.android_sdk.resolve()/'platforms'/('android-'+a.android_api)/'android.jar'
    if not android.is_file(): raise FileNotFoundError(android)
    jni=out/'jni017';jni.mkdir(exist_ok=True)
    classes=out/'classes';classes.mkdir(exist_ok=True)
    with zipfile.ZipFile(download/'litertlm.aar') as aar:
        (download/'classes.jar').write_bytes(aar.read('classes.jar'))
        (jni/'liblitertlm_jni.so').write_bytes(aar.read('jni/arm64-v8a/liblitertlm_jni.so'))
    ext='.exe' if os.name=='nt' else ''
    def java_tool(name): return str(a.jdk.resolve()/'bin'/(name+ext))
    jars=[download/'classes.jar']+[download/n for n in ARTIFACTS if n.endswith('.jar') and n!='r8.jar']
    source=pathlib.Path(__file__).parent/'android/DeviceRunnerJNI.java'
    def run(cmd):
        subprocess.run(cmd,check=True,timeout=180)
    run([java_tool('javac'),'--release','17','-cp',os.pathsep.join(map(str,[android,*jars])),
         '-d',str(classes),str(source)])
    run([java_tool('jar'),'--create','--file',str(out/'runner-java.jar'),'-C',str(classes),'.'])
    run([java_tool('java'),'-cp',str(download/'r8.jar'),'com.android.tools.r8.D8',
         '--release','--min-api','26','--lib',str(android),'--output',str(jni/'runner-dex.jar'),
         str(out/'runner-java.jar'),*map(str,jars)])
    wrapper='''#!/system/bin/sh
base="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export CLASSPATH="$base/jni017/runner-dex.jar"
if [ "$2" = "npu" ] && [ -n "$4" ]; then
  export LD_LIBRARY_PATH="$4${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ADSP_LIBRARY_PATH="$4;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp"
fi
exec /system/bin/app_process "-Djava.library.path=$base/jni017${4:+:$4}" /system/bin com.google.ai.edge.litertlm.DeviceRunnerJNI "$@"
'''
    (out/'device_runner_jni').write_text(wrapper,encoding='utf-8',newline='\n')
    report={'runtime':'litertlm-android-0.17.0','architecture':'arm64-v8a',
            'source_sha256':digest(source),'dependencies':ARTIFACTS,
            'outputs':{str(f.relative_to(out)):digest(f) for f in [jni/'liblitertlm_jni.so',jni/'runner-dex.jar',out/'device_runner_jni']}}
    (out/'build-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
