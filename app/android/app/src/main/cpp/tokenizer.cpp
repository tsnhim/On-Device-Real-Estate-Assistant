#include <jni.h>
#include <string>
#include <vector>
#include "sentencepiece_processor.h"

static sentencepiece::SentencePieceProcessor sp;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_coolcall_MainActivity_loadModel(JNIEnv* env, jobject thiz, jstring path) {
    const char* model_path = env->GetStringUTFChars(path, nullptr);
    auto status = sp.Load(model_path);
    env->ReleaseStringUTFChars(path, model_path);
    return status.ok();
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_example_coolcall_MainActivity_nativeEncode(JNIEnv* env, jobject thiz, jstring text) {
    const char* input_text = env->GetStringUTFChars(text, nullptr);
    std::vector<int> ids;
    sp.Encode(input_text, &ids);
    env->ReleaseStringUTFChars(text, input_text);

    jlongArray result = env->NewLongArray(ids.size());
    jlong* result_ptr = env->GetLongArrayElements(result, nullptr);
    for (size_t i = 0; i < ids.size(); ++i) {
        result_ptr[i] = static_cast<jlong>(ids[i]);
    }
    env->ReleaseLongArrayElements(result, result_ptr, 0);
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_coolcall_MainActivity_nativeDecode(JNIEnv* env, jobject thiz, jlongArray ids) {
    jsize len = env->GetArrayLength(ids);
    jlong* ids_ptr = env->GetLongArrayElements(ids, nullptr);
    std::vector<int> int_ids(len);
    for (int i = 0; i < len; ++i) {
        int_ids[i] = static_cast<int>(ids_ptr[i]);
    }
    env->ReleaseLongArrayElements(ids, ids_ptr, JNI_ABORT);

    std::string text;
    sp.Decode(int_ids, &text);
    return env->NewStringUTF(text.c_str());
}
