"""
Integration testing with the CLI

"""
from cli.cli import cli
from click.testing import CliRunner

def test_help():
    """Tests the command-line interface help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output

# Testing of the add_cli of the cli group
def test_add_cli():
    """Tests the command-line interface add command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["add", "1", "2"])
    assert result.exit_code == 0
    assert "3" in result.output


# Testing of the subtract_cli of the cli group
def test_subtract_cli():
    """Tests the command-line interface subtract command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["subtract", "5", "3"])
    assert result.exit_code == 0
    assert "2" in result.output


# Testing of the multiply_cli of the cli group
def test_multiply_cli():
    """Tests the command-line interface multiply command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["multiply", "2", "3"])
    assert result.exit_code == 0
    assert "6" in result.output


# Testing of the divide_cli of the cli group
def test_divide_cli():
    """Tests the command-line interface divide command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["divide", "6", "3"])
    assert result.exit_code == 0
    assert "2" in result.output


# Testing of the power_cli of the cli group
def test_power_cli():
    """Tests the command-line interface power command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["power", "2", "3"])
    assert result.exit_code == 0
    assert "8" in result.output
