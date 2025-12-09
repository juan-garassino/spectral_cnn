"""
Key Code Examples Extraction for Spectral GPT Documentation

This module extracts and formats the key code examples needed for
academic papers and documentation, including:
- WavePacketEmbedding implementation
- Interference attention code
- RGD optimizer code
- QFE loss code
- High-level API usage examples
"""

from pathlib import Path
from typing import Dict, List, Optional
from code_extractor import CodeExtractor, CodeSnippet


class SpectralGPTCodeExamples:
    """
    Extract and manage key code examples for Spectral GPT documentation.
    
    This class provides convenient access to all the important code snippets
    needed for papers and documentation.
    """
    
    def __init__(self, base_dir: str = "spectral_gpt"):
        """
        Initialize the code examples extractor.
        
        Args:
            base_dir: Base directory containing the spectral_gpt source code
        """
        self.base_dir = Path(base_dir)
        self.extractor = CodeExtractor()
        self._examples_cache: Dict[str, CodeSnippet] = {}
        
    def get_wave_packet_embedding(
        self,
        include_methods: Optional[List[str]] = None
    ) -> Optional[CodeSnippet]:
        """
        Extract the WavePacketEmbedding class implementation.
        
        Args:
            include_methods: Specific methods to include (None = all, ['forward'] = just forward)
            
        Returns:
            CodeSnippet for WavePacketEmbedding class
        """
        cache_key = f"wave_packet_embedding_{include_methods}"
        if cache_key in self._examples_cache:
            return self._examples_cache[cache_key]
            
        file_path = self.base_dir / "wave_gpt.py"
        snippet = self.extractor.extract_class(
            str(file_path),
            "WavePacketEmbedding",
            include_methods=include_methods
        )
        
        if snippet:
            self._examples_cache[cache_key] = snippet
            
        return snippet
    
    def get_interference_attention(
        self,
        include_methods: Optional[List[str]] = None
    ) -> Optional[CodeSnippet]:
        """
        Extract the interference attention mechanism implementation.
        
        Args:
            include_methods: Specific methods to include
            
        Returns:
            CodeSnippet for interference attention
        """
        cache_key = f"interference_attention_{include_methods}"
        if cache_key in self._examples_cache:
            return self._examples_cache[cache_key]
            
        file_path = self.base_dir / "wave_gpt.py"
        
        # Try to find WaveAttention or similar class
        # First, let's check what attention classes exist
        snippet = self.extractor.extract_class(
            str(file_path),
            "WaveAttention",
            include_methods=include_methods
        )
        
        if snippet:
            self._examples_cache[cache_key] = snippet
            
        return snippet
    
    def get_rgd_optimizer(
        self,
        include_methods: Optional[List[str]] = None
    ) -> Optional[CodeSnippet]:
        """
        Extract the Resonant Gradient Descent optimizer implementation.
        
        Args:
            include_methods: Specific methods to include (e.g., ['step', '__init__'])
            
        Returns:
            CodeSnippet for RGD optimizer
        """
        cache_key = f"rgd_optimizer_{include_methods}"
        if cache_key in self._examples_cache:
            return self._examples_cache[cache_key]
            
        file_path = self.base_dir / "physics_optim.py"
        snippet = self.extractor.extract_class(
            str(file_path),
            "ResonantGradientDescent",
            include_methods=include_methods
        )
        
        if snippet:
            self._examples_cache[cache_key] = snippet
            
        return snippet
    
    def get_qfe_loss(
        self,
        include_methods: Optional[List[str]] = None
    ) -> Optional[CodeSnippet]:
        """
        Extract the Quantum Field Entanglement loss implementation.
        
        Args:
            include_methods: Specific methods to include
            
        Returns:
            CodeSnippet for QFE loss
        """
        cache_key = f"qfe_loss_{include_methods}"
        if cache_key in self._examples_cache:
            return self._examples_cache[cache_key]
            
        file_path = self.base_dir / "physics_optim.py"
        snippet = self.extractor.extract_class(
            str(file_path),
            "QuantumFieldEntanglementLoss",
            include_methods=include_methods
        )
        
        if snippet:
            self._examples_cache[cache_key] = snippet
            
        return snippet
    
    def get_wave_gpt_config(self) -> Optional[CodeSnippet]:
        """
        Extract the WaveGPTConfig class.
        
        Returns:
            CodeSnippet for WaveGPTConfig
        """
        if "wave_gpt_config" in self._examples_cache:
            return self._examples_cache["wave_gpt_config"]
            
        file_path = self.base_dir / "wave_gpt.py"
        snippet = self.extractor.extract_class(
            str(file_path),
            "WaveGPTConfig"
        )
        
        if snippet:
            self._examples_cache["wave_gpt_config"] = snippet
            
        return snippet
    
    def get_api_usage_example(self) -> str:
        """
        Generate a high-level API usage example.
        
        Returns:
            String containing example code for using Spectral GPT
        """
        example = '''# High-Level API Usage Example

import torch
from spectral_gpt.wave_gpt import WaveGPT, WaveGPTConfig
from spectral_gpt.physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss

# 1. Configure the model
config = WaveGPTConfig(
    vocab_size=50257,
    block_size=256,
    d_model=384,
    num_layers=6,
    num_heads=6,
    num_waves=48,          # Number of wave packets per token
    num_harmonics=4,       # Harmonics per wave (1f, 2f, 3f, 4f)
    dropout=0.1
)

# 2. Create the model
model = WaveGPT(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Set up physics-informed optimization
optimizer = ResonantGradientDescent(
    model.parameters(),
    lr=3e-4,
    resonance_strength=0.3,  # Frequency-domain gradient filtering
    warmup_steps=500
)

# 4. Set up coherence loss
qfe_loss = QuantumFieldEntanglementLoss(
    lambda_coherence=0.1  # Weight for phase coherence term
)

# 5. Training loop
model.train()
for batch in dataloader:
    input_ids, targets = batch
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss with phase coherence
    loss = qfe_loss(logits, targets)
    
    # Backward pass with resonant gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# 6. Generation
model.eval()
prompt = torch.tensor([[1, 2, 3]])  # Token IDs
generated = model.generate(prompt, max_new_tokens=100, temperature=0.8)
'''
        return example
    
    def get_training_example(self) -> str:
        """
        Generate a complete training example.
        
        Returns:
            String containing example training code
        """
        example = '''# Complete Training Example

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spectral_gpt.wave_gpt import WaveGPT, WaveGPTConfig
from spectral_gpt.physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss

def train_spectral_gpt(
    train_data,
    val_data,
    vocab_size=50257,
    max_steps=10000,
    batch_size=32,
    learning_rate=3e-4
):
    """Train a Spectral GPT model with physics-informed optimization."""
    
    # Model configuration
    config = WaveGPTConfig(
        vocab_size=vocab_size,
        block_size=256,
        d_model=384,
        num_layers=6,
        num_heads=6,
        num_waves=48,
        num_harmonics=4,
        dropout=0.1
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveGPT(config).to(device)
    
    # Physics-informed optimizer
    optimizer = ResonantGradientDescent(
        model.parameters(),
        lr=learning_rate,
        resonance_strength=0.3,
        warmup_steps=500,
        weight_decay=0.01
    )
    
    # Coherence loss
    criterion = QuantumFieldEntanglementLoss(lambda_coherence=0.1)
    
    # Training loop
    model.train()
    for step in range(max_steps):
        # Get batch
        batch = next(iter(train_data))
        input_ids = batch['input_ids'].to(device)
        targets = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
        # Validation
        if step % 1000 == 0:
            model.eval()
            val_loss = evaluate(model, val_data, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")
            model.train()
    
    return model

def evaluate(model, val_data, criterion, device):
    """Evaluate model on validation data."""
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_data:
            input_ids = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches
'''
        return example
    
    def get_all_examples(self) -> Dict[str, str]:
        """
        Get all key code examples formatted for markdown.
        
        Returns:
            Dictionary mapping example names to formatted markdown strings
        """
        examples = {}
        
        # WavePacketEmbedding
        snippet = self.get_wave_packet_embedding(include_methods=['forward'])
        if snippet:
            examples['wave_packet_embedding'] = self.extractor.format_for_markdown(
                snippet,
                max_lines=50
            )
        
        # Interference Attention
        snippet = self.get_interference_attention(include_methods=['forward'])
        if snippet:
            examples['interference_attention'] = self.extractor.format_for_markdown(
                snippet,
                max_lines=40
            )
        
        # RGD Optimizer
        snippet = self.get_rgd_optimizer(include_methods=['step'])
        if snippet:
            examples['rgd_optimizer'] = self.extractor.format_for_markdown(
                snippet,
                max_lines=60
            )
        
        # QFE Loss
        snippet = self.get_qfe_loss(include_methods=['forward'])
        if snippet:
            examples['qfe_loss'] = self.extractor.format_for_markdown(
                snippet,
                max_lines=40
            )
        
        # API Usage
        examples['api_usage'] = f"```python\n{self.get_api_usage_example()}\n```"
        
        # Training Example
        examples['training_example'] = f"```python\n{self.get_training_example()}\n```"
        
        return examples
    
    def generate_code_appendix(self, output_file: Optional[str] = None) -> str:
        """
        Generate a complete code appendix for a paper.
        
        Args:
            output_file: Optional file path to save the appendix
            
        Returns:
            Markdown string containing the code appendix
        """
        md = "# Appendix: Code Examples\n\n"
        md += "This appendix provides key implementation details for reproducibility.\n\n"
        
        # Section 1: Wave Packet Embedding
        md += "## A.1 Wave Packet Embedding\n\n"
        md += "The core innovation of Spectral GPT is representing tokens as wave packets "
        md += "with learnable frequencies, phases, and harmonic amplitudes.\n\n"
        
        snippet = self.get_wave_packet_embedding(include_methods=['__init__', 'forward'])
        if snippet:
            md += self.extractor.format_for_markdown(snippet, max_lines=80)
            md += "\n\n"
        
        # Section 2: Interference Attention
        md += "## A.2 Interference Attention\n\n"
        md += "Instead of dot-product attention, we use wave interference to compute "
        md += "attention weights based on phase relationships.\n\n"
        
        snippet = self.get_interference_attention(include_methods=['forward'])
        if snippet:
            md += self.extractor.format_for_markdown(snippet, max_lines=60)
            md += "\n\n"
        
        # Section 3: Resonant Gradient Descent
        md += "## A.3 Resonant Gradient Descent (RGD)\n\n"
        md += "Physics-informed optimizer that filters gradients in the frequency domain, "
        md += "applying stronger updates at resonant frequencies.\n\n"
        
        snippet = self.get_rgd_optimizer(include_methods=['__init__', 'step'])
        if snippet:
            md += self.extractor.format_for_markdown(snippet, max_lines=80)
            md += "\n\n"
        
        # Section 4: Quantum Field Entanglement Loss
        md += "## A.4 Quantum Field Entanglement (QFE) Loss\n\n"
        md += "Coherence loss that enforces phase alignment between predictions and targets "
        md += "in the frequency domain.\n\n"
        
        snippet = self.get_qfe_loss(include_methods=['forward'])
        if snippet:
            md += self.extractor.format_for_markdown(snippet, max_lines=60)
            md += "\n\n"
        
        # Section 5: High-Level API
        md += "## A.5 High-Level API Usage\n\n"
        md += "Example of using the Spectral GPT API for training and inference.\n\n"
        md += f"```python\n{self.get_api_usage_example()}\n```\n\n"
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"Code appendix saved to: {output_path}")
        
        return md


def main():
    """Demo the code extraction functionality."""
    print("🔍 Extracting Spectral GPT Code Examples...\n")
    
    extractor = SpectralGPTCodeExamples()
    
    # Extract individual examples
    print("1. Wave Packet Embedding:")
    snippet = extractor.get_wave_packet_embedding(include_methods=['forward'])
    if snippet:
        print(f"   ✓ Extracted {snippet.name} ({snippet.end_line - snippet.start_line} lines)")
    else:
        print("   ✗ Not found")
    
    print("\n2. RGD Optimizer:")
    snippet = extractor.get_rgd_optimizer(include_methods=['step'])
    if snippet:
        print(f"   ✓ Extracted {snippet.name} ({snippet.end_line - snippet.start_line} lines)")
    else:
        print("   ✗ Not found")
    
    print("\n3. QFE Loss:")
    snippet = extractor.get_qfe_loss(include_methods=['forward'])
    if snippet:
        print(f"   ✓ Extracted {snippet.name} ({snippet.end_line - snippet.start_line} lines)")
    else:
        print("   ✗ Not found")
    
    # Generate complete appendix
    print("\n4. Generating Code Appendix...")
    appendix = extractor.generate_code_appendix(
        output_file="experiments/paper/code_appendix.md"
    )
    print(f"   ✓ Generated appendix ({len(appendix)} characters)")
    
    print("\n✅ Code extraction complete!")


if __name__ == "__main__":
    main()
