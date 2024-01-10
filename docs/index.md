<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://github.com/kyegomez/zeta/raw/master/images/zeta.png"
      >
    </a>
  </p>
</div>

## 👋 Hello

zeta provides you with all the building blocks you need to build reliable, production-grade, and scalable multi-agent apps!

## 💻 Install

You can install `zeta` with pip in a
[**Python>=3.8**](https://www.python.org/) environment.

!!! example "pip install (recommended)"

    === "headless"
        The headless installation of `zeta` is designed for environments where graphical user interfaces (GUI) are not needed, making it more lightweight and suitable for server-side applications.

        ```bash
        pip install zeta
        ```


!!! example "git clone (for development)"

    === "virtualenv"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/kyegomez/zeta.git
        cd zeta

        # setup python environment and activate it
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip

        # headless install
        pip install -e "."

        # desktop install
        pip install -e ".[desktop]"
        ```

    === "poetry"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/kyegomez/zeta.git
        cd zeta

        # setup python environment and activate it
        poetry env use python3.10
        poetry shell

        # headless install
        poetry install

        # desktop install
        poetry install --extras "desktop"
        ```


## Documentation

[Learn more about zeta →](zeta/)


## Examples

Check out zeta examples for building agents, data retrieval, and more.

[Checkout zeta examples →](examples/)
