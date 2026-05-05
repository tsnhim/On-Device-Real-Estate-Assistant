package com.example.coolcall

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ai.onnxruntime.*
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.background
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Mic

data class ChatMessage(val text: String, val isUser: Boolean)

class MainActivity : ComponentActivity() {

    private val _messages = mutableStateListOf<ChatMessage>()
    val messages: List<ChatMessage> get() = _messages

    private var _isProcessing = mutableStateOf(false)
    val isProcessing: Boolean get() = _isProcessing.value

    lateinit var isListening: MutableState<Boolean>

    lateinit var env: OrtEnvironment
    lateinit var encoderSession: OrtSession
    lateinit var decoderSession: OrtSession

    external fun loadModel(path: String): Boolean
    external fun nativeEncode(text: String): LongArray
    external fun nativeDecode(ids: LongArray): String

    companion object {
        init {
            System.loadLibrary("coolcall")
        }
    }

    val speechLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        isListening.value = false
        if (result.resultCode == Activity.RESULT_OK) {
            val text = result.data
                ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                ?.get(0)
            if (text != null) {
                _messages.add(ChatMessage(text, true))
                runOnnx(text)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        isListening = mutableStateOf(false)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val modelDir = filesDir.absolutePath + "/onnx_model"
                copyAssetFolder(assets, "onnx_model", modelDir)

                env = OrtEnvironment.getEnvironment()
                encoderSession = env.createSession("$modelDir/encoder_model.onnx")
                decoderSession = env.createSession("$modelDir/decoder_model.onnx")

                val spModelPath = "$modelDir/spiece.model"
                if (!loadModel(spModelPath)) {
                    throw Exception("Failed to load SentencePiece model at $spModelPath")
                }

                withContext(Dispatchers.Main) {
                    _messages.add(ChatMessage("Ready to chat!", false))
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    _messages.add(ChatMessage("Error: ${e.message}", false))
                }
            }
        }

        setContent { MyApp() }
    }

    fun copyAssetFolder(assetManager: android.content.res.AssetManager, from: String, to: String) {
        val files = assetManager.list(from) ?: return
        val dir = java.io.File(to)
        if (!dir.exists()) dir.mkdirs()

        for (file in files) {
            val fullFrom = "$from/$file"
            val fullTo = "$to/$file"
            val subFiles = assetManager.list(fullFrom)
            if (subFiles != null && subFiles.isNotEmpty()) {
                copyAssetFolder(assetManager, fullFrom, fullTo)
            } else {
                assetManager.open(fullFrom).use { input ->
                    java.io.FileOutputStream(fullTo).use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    fun runOnnx(input: String) {
        _isProcessing.value = true
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // 1. Use the correct prompt from your Python code
                val prompt = "You are a real estate expert.\nQuestion: $input"
                val tokens = nativeEncode(prompt)

                val inputTensor = OnnxTensor.createTensor(
                    env,
                    java.nio.LongBuffer.wrap(tokens),
                    longArrayOf(1, tokens.size.toLong())
                )

                val mask = LongArray(tokens.size) { 1 }
                val maskTensor = OnnxTensor.createTensor(
                    env,
                    java.nio.LongBuffer.wrap(mask),
                    longArrayOf(1, mask.size.toLong())
                )

                val encoderOut = encoderSession.run(
                    mapOf("input_ids" to inputTensor, "attention_mask" to maskTensor)
                )

                val encoderTensor = encoderOut[0] as OnnxTensor

                var decoderInput = longArrayOf(0)
                val outputTokens = mutableListOf<Int>()
                val repetitionPenalty = 4.0f

                repeat(500) {

                    val decTensor = OnnxTensor.createTensor(
                        env,
                        java.nio.LongBuffer.wrap(decoderInput),
                        longArrayOf(1, decoderInput.size.toLong())
                    )

                    val outputs = decoderSession.run(
                        mapOf(
                            "input_ids" to decTensor,
                            "encoder_hidden_states" to encoderTensor,
                            "encoder_attention_mask" to maskTensor
                        )
                    )

                    val logits = outputs[0].value as Array<Array<FloatArray>>
                    val lastLogits = logits[0][decoderInput.size - 1].copyOf()

                    // 2. Apply Repetition Penalty to prevent loops
                    for (id in outputTokens) {
                        if (lastLogits[id] > 0) {
                            lastLogits[id] /= repetitionPenalty
                        } else {
                            lastLogits[id] *= repetitionPenalty
                        }
                    }

                    val next = lastLogits.indices.maxBy { lastLogits[it] }
                    
                    decTensor.close()
                    outputs.close()

                    if (next == 1) return@repeat // 1 is EOS
                    
                    outputTokens.add(next)
                    decoderInput += next.toLong()
                }

                val text = nativeDecode(outputTokens.map { it.toLong() }.toLongArray())
                val finalResult = if (text.isEmpty()) "Model produced no output" else text
                
                withContext(Dispatchers.Main) {
                    _isProcessing.value = false
                    _messages.add(ChatMessage(finalResult, false))
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    _isProcessing.value = false
                    _messages.add(ChatMessage("Error: ${e.message}", false))
                }
            }
        }
    }

    @Composable
    fun MyApp() {
        val listState = rememberLazyListState()
        var textInput by remember { mutableStateOf("") }
        
        LaunchedEffect(messages.size, isProcessing) {
            val totalItems = messages.size + if (isProcessing) 1 else 0
            if (totalItems > 0) {
                listState.animateScrollToItem(totalItems - 1)
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            Text(
                "CoolCall Chat",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                reverseLayout = false
            ) {
                items(messages) { message ->
                    ChatBubble(message)
                }
                if (isProcessing) {
                    item {
                        ChatBubble(ChatMessage("thinking...", false))
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextField(
                    value = textInput,
                    onValueChange = { textInput = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Type a message...") },
                    shape = RoundedCornerShape(24.dp),
                    colors = TextFieldDefaults.colors(
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        disabledIndicatorColor = Color.Transparent,
                        errorIndicatorColor = Color.Transparent
                    ),
                    singleLine = true
                )

                Spacer(Modifier.width(8.dp))

                if (textInput.isEmpty()) {
                    IconButton(
                        onClick = {
                            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                            }
                            speechLauncher.launch(intent)
                        },
                        modifier = Modifier
                            .size(48.dp)
                            .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(24.dp))
                    ) {
                        Icon(Icons.Default.Mic, contentDescription = "Speak", tint = Color.White)
                    }
                } else {
                    IconButton(
                        onClick = {
                            if (textInput.isNotBlank()) {
                                val message = textInput
                                _messages.add(ChatMessage(message, true))
                                runOnnx(message)
                                textInput = ""
                            }
                        },
                        modifier = Modifier
                            .size(48.dp)
                            .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(24.dp))
                    ) {
                        Icon(Icons.Default.Send, contentDescription = "Send", tint = Color.White)
                    }
                }
            }
        }
    }

    @Composable
    fun ChatBubble(message: ChatMessage) {
        val alignment = if (message.isUser) Alignment.CenterEnd else Alignment.CenterStart
        val bgColor = if (message.isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
        val textColor = if (message.isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp),
            contentAlignment = alignment
        ) {
            Surface(
                color = bgColor,
                shape = RoundedCornerShape(
                    topStart = 16.dp,
                    topEnd = 16.dp,
                    bottomStart = if (message.isUser) 16.dp else 0.dp,
                    bottomEnd = if (message.isUser) 0.dp else 16.dp
                ),
                tonalElevation = 2.dp
            ) {
                Text(
                    text = message.text,
                    color = textColor,
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }
    }
}